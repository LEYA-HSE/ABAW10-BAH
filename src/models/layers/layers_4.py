import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian, add_remaining_self_loops, add_self_loops

from torch_scatter import scatter_add

from math import ceil


class FixedPointLayer(nn.Module):
    def __init__(self, channels, tol=1e-5, max_iter=100):
        super(FixedPointLayer, self).__init__()

        self.tol = tol
        self.max_iter = max_iter

        self.lin = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        z = torch.zeros_like(x)
        for i in range(self.max_iter):
            z_next = torch.tanh(self.lin(z) + x)
            error = torch.norm(z - z_next)
            z = z_next
            if error < self.tol:
                break
        return z


class WaveLapConv(MessagePassing):
    def __init__(self, channels, dropout,
                 laplacian='sym', method='exp', dt=1., init_residual=False, exp_init=False):
        super(WaveLapConv, self).__init__()

        self.channels = channels
        self.dropout = dropout

        self.laplacian = laplacian
        self.method = method
        self.dt = dt
        self.init_residual = init_residual
        self.exp_init = exp_init

        # temp
        self.eps_ = nn.Parameter(torch.Tensor(1))

        if laplacian == 'fa':
            self.att_j = nn.Linear(channels, 1, bias=False)
            self.att_i = nn.Linear(channels, 1, bias=False)

        if method == 'exp':
            self.scheme = self.explicit_propagate
        elif method == 'imp':
            self.scheme = self.implicit_propagate
            self.fixed_point = FixedPointLayer(channels)
        else:
            raise NotImplementedError

        if self.init_residual:
            self.eps = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.laplacian == 'fa':
            self.att_j.reset_parameters()
            self.att_i.reset_parameters()

        if self.init_residual:
            uniform_(self.eps)

    def forward(self, x, x_pre, edge_index, x_0=None):
        edge_index, edge_weight = self.get_laplacian(x, edge_index)
        x = self.scheme(x, x_pre, edge_index, edge_weight)

        if self.init_residual and x_0 is not None:
            x = x + self.eps * x_0

        return x

    def get_laplacian(self, x, edge_index):
        edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=edge_index.device)
        num_nodes = x.size(0)

        if self.laplacian == 'sym':
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0., num_nodes)
            edge_index, edge_weight = self.normalize_laplacian(edge_index, edge_weight, num_nodes)

        elif self.laplacian == 'fa':
            edge_index, edge_weight = gcn_norm(edge_index, None, num_nodes, dtype=x.dtype)

            alpha_j = self.att_j(x)[edge_index[0]]
            alpha_i = self.att_i(x)[edge_index[1]]

            alpha = torch.tanh(alpha_j + alpha_i).squeeze(-1)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            edge_weight = alpha * edge_weight

        else:
            raise NotImplementedError

        return edge_index, edge_weight

    @staticmethod
    def normalize_laplacian(edge_index, edge_weight, num_nodes):
        row, col = edge_index[0], edge_index[1]
        idx = col
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    def explicit_propagate(self, x, x_pre, edge_index, edge_weight):
        num_nodes = x.size(0)

        if self.exp_init:
            edge_weight = self.dt ** 2 * edge_weight / 2
            edge_weight[-num_nodes:] += 1

            x = self.dt * x + self.propagate(edge_index, x=x_pre, edge_weight=edge_weight)

        else:
            edge_weight = self.dt ** 2 * edge_weight
            edge_weight[-num_nodes:] += 2

            x = self.propagate(edge_index, x=x, edge_weight=edge_weight) - x_pre

        return x

    def implicit_propagate(self, x, x_pre, edge_index, edge_weight):
        num_nodes = x.size(0)

        edge_weight_b = self.dt ** 2 * edge_weight / 2
        edge_weight_b[-num_nodes:] += 2

        edge_weight_c = self.dt ** 2 * edge_weight / 4
        edge_weight_c[-num_nodes:] -= 1

        x = self.propagate(edge_index, x=x, edge_weight=edge_weight_b) + \
            self.propagate(edge_index, x=x_pre, edge_weight=edge_weight_c)

        x = self.fixed_point(x)

        return x

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}' \
               f'(channels={self.channels}, dt={self.dt}, laplacian={self.laplacian}, method={self.method})'



class WavePDEFunc(nn.Module):
    def __init__(self, dataset, time, dim_hidden, dropout,
                 laplacian='sym', method='exp', dt=1., K=1, init_residual=False, **kwargs):
        super(WavePDEFunc, self).__init__()

        self.dim_hidden = dim_hidden

        self.num_classes = dataset.num_classes
        self.n = ceil(time / dt)
        self.method = method
        self.init_residual = init_residual

        # HyperbolicGNN convolutional layers
        self.convs = nn.ModuleList()
        for tn in range(self.n - 1):
            if laplacian in ['sym', 'fa']:
                self.convs.append(WaveLapConv(dim_hidden, dropout, laplacian, method, dt, init_residual))


        if method == 'exp':
            if laplacian in ['sym', 'fa']:
                self.exp_conv_0 = WaveLapConv(dim_hidden, dropout, laplacian, method, dt, init_residual, exp_init=True)


        # Linear layers
        self.lin0 = nn.Linear(dataset.num_features, dim_hidden)
        self.lin1 = nn.Linear(dataset.num_features, dim_hidden)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.method == 'exp':
            self.exp_conv_0.reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()

    def set_init_value(self, x, edge_index):
        phi_0 = self.lin0(x)
        phi_1 = self.lin1(x)

        if self.method == 'exp':
            x_1 = self.exp_conv_0(phi_1, phi_0, edge_index)
        elif self.method == 'imp':
            x_1 = self.dt * phi_1 * phi_0
        else:
            raise NotImplementedError

        return phi_0, x_1

    def forward(self, x, edge_index):
        # Set initial value
        x_0, x_1 = self.set_init_value(x, edge_index)

        # Solve the initial value problem of graph wave equation
        x, x_pre = x_1, x_0
        for tn in range(self.n - 1):
            if not self.init_residual:
                x_next = self.convs[tn](x, x_pre, edge_index)
            else:
                x_next = self.convs[tn](x, x_pre, edge_index, x_0=x_0)

            x_pre = x
            x = x_next

        return x


class _MiniDataset:
    def __init__(self, num_features: int, num_classes: int, name: str = "Custom"):
        self.num_features = num_features
        self.num_classes = num_classes
        self.name = name

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int | None = None, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim
        self.dropout = dropout
        self.alpha = alpha

        time = 3.0
        dt = 1.0
        K = 1
        laplacian = "fa"
        method = "exp"
        init_residual = False
        dataset_name = "MM"

        ds = _MiniDataset(num_features=self.in_dim, num_classes=self.out_dim, name=dataset_name)

        self.pde = WavePDEFunc(
            dataset=ds,
            time=time,
            dim_hidden=self.out_dim,
            dropout=self.dropout,
            laplacian=laplacian,
            method=method,
            dt=dt,
            K=K,
            init_residual=init_residual,
            dataset_name=dataset_name
        )

    @staticmethod
    def _dense_adj_to_edge_index(adj_b: torch.Tensor) -> torch.Tensor:

        idx = adj_b.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=adj_b.device)
        return idx.t().contiguous().long()

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        B, N, D = h.shape

        outs = []
        for b in range(B):
            x_b = h[b]
            edge_index_b = self._dense_adj_to_edge_index(adj[b])
            out_b = self.pde(x_b, edge_index_b)
            outs.append(out_b)
        return torch.stack(outs, dim=0)

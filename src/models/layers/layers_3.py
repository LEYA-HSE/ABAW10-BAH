from torch_geometric.data import Data, Batch
import torch
import torch.nn as nn
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import typing
from typing import Callable, Optional, Union, Tuple, List
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn.inits import reset, glorot, zeros
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    to_undirected,
)

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_sparse import SparseTensor as TSSparseTensor

from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    Size,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


class GroupSort(nn.Module):
    def forward(self, x):
        a, b = x.split(x.size(-1) // 2, 1)
        a, b = torch.max(a, b), torch.min(a, b)
        return torch.cat([a, b], dim=-1)

def block_diagonal_init(weight_matrix, block_size=2, bound=0.5):

    n = weight_matrix.size(0)

    weight_matrix = torch.zeros_like(weight_matrix)

    num_blocks = (n + block_size - 1) // block_size

    for i in range(num_blocks):
        actual_block_size = min(block_size, n - i * block_size)

        part = torch.randn(actual_block_size, actual_block_size) * bound

        start_row = i * block_size
        end_row = start_row + actual_block_size
        weight_matrix[start_row:end_row, start_row:end_row] = part

    return weight_matrix


class OrthogonalGCNConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, residual=False,
                 global_bias=True, T=10, use_hermitian=False,
                 activation=torch.nn.ReLU,
                 **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        if global_bias:
            self.bias = torch.nn.Parameter(torch.zeros(dim_out))
        else:
            self.register_parameter('bias', None)

        if use_hermitian:
            base_conv = HermitianGCNConv
        else:
            base_conv = ComplexToRealGCNConv

        self.act = nn.Sequential(
            activation(),
            nn.Dropout(self.dropout),
        )
        self.model = TaylorGCNConv(base_conv(dim_in, dim_out, **kwargs), T=T)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index)
        if self.bias is not None:
            batch.x = batch.x + self.bias
        batch.x = self.act(batch.x)

        if self.residual:
            batch.x = x_in + batch.x

        return batch


class HermitianGCNConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: Optional[bool] = False,
            normalize: bool = True,
            bias: bool = False,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        if in_channels != out_channels:
            raise ValueError(
                f"Input and output dimension must be the same for HermitianGCNConv. Got in_channels = {in_channels}, out_channels = {out_channels}")
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin.weight.upper_triangular_params = in_channels * (
                    in_channels - 1) // 2

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.weight.data = block_diagonal_init(self.lin.weight.data)
        self._cached_edge_index = None
        self._cached_adj_t = None
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None,
                apply_feature_lin: bool = True,
                return_feature_only: bool = False) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if apply_feature_lin:
            if return_feature_only:
                return x

        x = (x @ self.lin.weight - x @ self.lin.weight.T) / 2
        if self.bias is not None:
            x = x + self.bias

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class ComplexToRealGCNConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: Optional[bool] = False,
            normalize: bool = True,
            bias: bool = False,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None


        if out_channels % 2 != 0:
            raise ValueError(
                f"Output dimension must be even for ComplexToRealGCNConv. Got out_channels = {out_channels}")

        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

        multiplier = torch.ones(out_channels)
        multiplier[out_channels // 2:] = -1
        self.register_buffer('multiplier', multiplier.view(1, -1))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels // 2))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None,
                apply_feature_lin: bool = True,
                return_feature_only: bool = False) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if apply_feature_lin:
            x = self.lin(x)
            if self.bias is not None:
                x = x + self.bias
            if return_feature_only:
                return x

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out * self.multiplier
        out = torch.roll(out, self.out_channels // 2, -1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class TaylorGCNConv(MessagePassing):
    def __init__(
            self,
            conv: HermitianGCNConv,
            T: int = 16,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.conv = conv
        self.T = T

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, **kwargs) -> Tensor:
        c = 1.
        x = self.conv(x, edge_index, edge_weight,
                      apply_feature_lin=True,
                      return_feature_only=True)
        x_k = x.clone()  # Create a copy of the input tensor

        for k in range(self.T):
            x_k = self.conv(x_k, edge_index, edge_weight, apply_feature_lin=False, **kwargs) / (k + 1)
            x += x_k
        return x


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(  # noqa: F811
        edge_index: Adj,
        edge_weight: OptTensor = None,
        num_nodes: Optional[int] = None,
        improved: bool = False,
        add_self_loops: bool = True,
        flow: str = "source_to_target",
        dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class _UnitaryGALAdapter(nn.Module):

    def __init__(self,
                 in_dim: int,
                 out_dim: int = None,
                 dropout: float = 0.0,
                 use_hermitian: bool = False,
                 T: int = 10,
                 residual: bool = True,
                 use_edge_weight: bool = False):
        super().__init__()
        out_dim = out_dim or in_dim

        if not use_hermitian and (out_dim % 2 != 0):
            raise ValueError("For ComplexToReal (use_hermitian=False) out_dim must be even.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_edge_weight = use_edge_weight

        self.unitary = OrthogonalGCNConvLayer(
            dim_in=in_dim,
            dim_out=out_dim,
            dropout=dropout,
            residual=residual,
            global_bias=True,
            T=T,
            use_hermitian=use_hermitian,
        )

    @staticmethod
    def _dense_adj_to_pyg_batch(x_bnd: torch.Tensor,
                                adj_bnn: torch.Tensor,
                                use_edge_weight: bool) -> Batch:

        B, N, D = x_bnd.shape
        data_list = []
        for b in range(B):
            x = x_bnd[b]  # [N,D]

            mask = adj_bnn[b] != 0
            mask = mask.clone()
            mask.fill_diagonal_(False)

            idx = mask.nonzero(as_tuple=False)
            if idx.numel() == 0:
                data_list.append(Data(x=x, num_nodes=N))
                continue

            edge_index = idx.t().contiguous().long()

            if use_edge_weight:
                edge_attr = adj_bnn[b][edge_index[0], edge_index[1]].to(x.dtype)
                data_list.append(Data(x=x, edge_index=edge_index,
                                      edge_attr=edge_attr, num_nodes=N))
            else:
                data_list.append(Data(x=x, edge_index=edge_index, num_nodes=N))

        return Batch.from_data_list(data_list)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        B, N, D = h.shape
        device = h.device

        batch = self._dense_adj_to_pyg_batch(h, adj, self.use_edge_weight).to(device)

        if self.use_edge_weight and (batch.edge_index is not None):
            row, col = batch.edge_index
            if getattr(batch, "edge_attr", None) is not None:
                val = batch.edge_attr.to(h.dtype)
            else:

                val = torch.ones(row.size(0), dtype=h.dtype, device=row.device)

            num_nodes = batch.x.size(0)
            batch.edge_index = TSSparseTensor(
                row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes)
            )

            if hasattr(batch, "edge_attr"):
                batch.edge_attr = None

        batch = self.unitary(batch)

        out = batch.x.view(B, N, self.out_dim)
        return out


class GraphAttentionLayer(_UnitaryGALAdapter):

    def __init__(self, in_dim, out_dim=None, dropout=0.2, alpha=0.2):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim or in_dim,
            dropout=0.2,
            use_hermitian=False,
            T=10,
            residual=False,
            use_edge_weight=True,
        )

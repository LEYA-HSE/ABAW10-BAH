import random
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def random_walk(sample, num_nodes, edge_index, walk_len):

    device = edge_index.device

    src_nodes = edge_index[0]
    des_nodes = edge_index[1]
    nodes = torch.arange(num_nodes, device=device, dtype=torch.long).repeat(sample)

    out_neig = {int(node): [] for node in range(num_nodes)}
    for i in range(edge_index.size(-1)):
        out_neig[int(src_nodes[i].item())].append(des_nodes[i])

    total_samples = num_nodes * sample

    walks = -1 * torch.ones((walk_len, total_samples, 1), dtype=torch.long, device=device)

    new_seed = int.from_bytes(os.urandom(4), "little") ^ int(time.time() * 1e6)
    random.seed(new_seed)

    for i in range(total_samples):
        walks[0, i, 0] = nodes[i]
        current_node = walks[0, i, 0]
        for j in range(1, walk_len):
            neig = out_neig[int(current_node.item())]
            if len(neig) > 0:

                current_node = random.choice(neig)
                walks[j, i, 0] = current_node

    random.seed(42)
    return walks


def uniqueness(walks, total_samples):

    unique_walks = torch.empty_like(walks)
    for i in range(total_samples):
        c = 0
        om = {}
        for j, current_node in enumerate(walks[:, i, 0]):
            v = int(current_node.item())
            if v not in om:
                om[v] = c
                c += 1
            unique_walks[j, i, 0] = om[v]
    return unique_walks


class RumLayer(nn.Module):
    def __init__(
        self,
        num_nodes,
        sample,
        x_input_dim,
        hidden_state_dim,
        walk_len,
        rnd_walk: callable = random_walk,
        uniq_walk=uniqueness,
        **kwargs,
    ):
        super().__init__()
        self.rnn_walk = nn.GRU(2, hidden_state_dim, bidirectional=True)
        self.rnn_x = nn.GRU(x_input_dim, hidden_state_dim)

        self.num_nodes = num_nodes
        self.sample = sample
        self.x_input_dim = x_input_dim
        self.hidden_state_dim = hidden_state_dim
        self.total_samples = sample * num_nodes
        self.random_walk = rnd_walk
        self.uniq_walks = uniq_walk
        self.walk_len = walk_len

    def forward(self, x, edge_index):
        walks = self.random_walk(self.sample, self.num_nodes, edge_index, self.walk_len)

        x = x[walks.squeeze(-1)]

        uniq_walks = self.uniq_walks(walks, self.total_samples)
        uniq_walks = uniq_walks / uniq_walks.size(0)
        uniq_walks = uniq_walks * torch.pi * 2

        uniq_walks_sin, uniq_walks_cos = torch.sin(uniq_walks), torch.cos(uniq_walks)
        uniq_walks = torch.cat([uniq_walks_sin, uniq_walks_cos], dim=-1)

        _, h_walk = self.rnn_walk(uniq_walks)
        h_walk = torch.mean(h_walk, dim=0, keepdim=True)

        _, h = self.rnn_x(x, h_walk)
        h = h.view(self.sample, self.num_nodes, self.hidden_state_dim)
        h = torch.mean(h, dim=0)

        return h


class GraphAttentionLayer(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        dropout: float = 0.1,
        alpha: float = 0.2,
        *,
        sample: int = 2,
        walk_len: int = 4,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        self._rum_by_n = nn.ModuleDict()
        self._cfg = dict(sample=sample, walk_len=walk_len,
                         x_input_dim=in_dim, hidden_state_dim=out_dim)

    def _get_or_build_rum(self, n_nodes: int) -> nn.Module:
        key = str(n_nodes)
        if key not in self._rum_by_n:
            self._rum_by_n[key] = RumLayer(
                num_nodes=n_nodes,
                sample=self._cfg["sample"],
                x_input_dim=self._cfg["x_input_dim"],
                hidden_state_dim=self._cfg["hidden_state_dim"],
                walk_len=self._cfg["walk_len"],
            )
        return self._rum_by_n[key]

    @torch.no_grad()
    def _adj_to_edge_index(self, adj_b: torch.Tensor) -> torch.Tensor:

        idx = (adj_b > 0).nonzero(as_tuple=False)
        if idx.numel() == 0:
            n = adj_b.size(0)
            idx = torch.arange(n, device=adj_b.device)
            idx = torch.stack([idx, idx], dim=1)
        return idx.t().contiguous().long()

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        device = h.device
        B, N, Din = h.shape
        assert Din == self.in_dim, f"Expected in_dim={self.in_dim}, got {Din}"

        out_list = []

        for key in self._rum_by_n:
            self._rum_by_n[key].to(device)

        for b in range(B):
            x_bn = h[b]
            adj_bn = adj[b]
            edge_index = self._adj_to_edge_index(adj_bn)

            rum = self._get_or_build_rum(N).to(device)
            out_bn = rum(x_bn, edge_index)

            out_list.append(out_bn)

        out = torch.stack(out_list, dim=0)
        out = self.dropout(out)
        return out

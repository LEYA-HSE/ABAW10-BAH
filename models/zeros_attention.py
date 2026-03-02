import math
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import functional as F


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, transpose=False):
    return x * cos + rotate_half(x) * sin if not transpose else x * cos - rotate_half(x) * sin


def get_rotary_embedding(L, D, base=10000, device="cpu"):
    assert D % 2 == 0
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=device).float() / D))
    pos = torch.arange(L, device=device, dtype=torch.float32)
    s = pos[:, None] * inv[None]
    cos, sin = (t.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2) for t in (torch.cos(s), torch.sin(s)))
    return cos, sin


def compute_o(
    q: torch.Tensor,
    s_i: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    mask=None,
    causal=True,
    associative=True,
    is_first_layer=False,
    eps: float = 1e-12,
) -> torch.Tensor:
    B, L, h, d = q.shape

    if associative:
        t = torch.arange(1, L + 1, device=q.device, dtype=q.dtype).view(1, L, 1, 1) if causal else L

        s_i_max = s_i.max(dim=1, keepdim=True)[0].detach()
        s_i_stable = s_i - s_i_max

        exp_s_i = torch.exp(s_i_stable)

        if causal:
            E_t = exp_s_i.cumsum(dim=1)
            P_t = s_i.cumsum(dim=1)
        else:
            E_t = exp_s_i.sum(dim=1, keepdim=True).expand(-1, L, -1, -1)
            P_t = s_i.sum(dim=1, keepdim=True).expand(-1, L, -1, -1)

        kv = torch.einsum("blhd,blhe->blhde", k, v)

        if causal:
            F_t = (exp_s_i.unsqueeze(-1) * kv).cumsum(dim=1)
            G_t = (s_i.unsqueeze(-1) * kv).cumsum(dim=1)
            H_t = kv.cumsum(dim=1)
        else:
            F_t = (exp_s_i.unsqueeze(-1) * kv).sum(dim=1, keepdim=True).expand(-1, L, -1, -1, -1)
            G_t = (s_i.unsqueeze(-1) * kv).sum(dim=1, keepdim=True).expand(-1, L, -1, -1, -1)
            H_t = kv.sum(dim=1, keepdim=True).expand(-1, L, -1, -1, -1)

        sigma_t_1 = torch.sigmoid(gate[:, :, :, [0]])
        sigma_t_h = torch.sigmoid(gate[:, :, :, [1]])

        alpha_t = sigma_t_h / (E_t + eps)
        beta_t = (sigma_t_1 - sigma_t_h) / t
        gamma_t = -((beta_t * P_t + sigma_t_h) / t)

        out = torch.einsum(
            "blhd,blhde->blhe",
            q,
            alpha_t.unsqueeze(-1) * F_t
            + beta_t.unsqueeze(-1) * G_t
            + gamma_t.unsqueeze(-1) * H_t,
        )

        if is_first_layer:
            sigma_t_0 = torch.tanh(gate[:, :, :, [2]])
            zero_order = torch.einsum("blhd,blhde->blhe", q / t, H_t) * sigma_t_0
            out = out + zero_order

    else:
        if causal:
            if mask is None:
                base = torch.tril(torch.ones((L, L), device=q.device, dtype=torch.bool))
                mask = base.unsqueeze(0).unsqueeze(0)

        t = torch.arange(1, L + 1, device=q.device, dtype=q.dtype).view(1, 1, L, 1) if causal else int(L)
        cos_theta = torch.einsum("blhd,bihd->bhli", q, k)
        cos_theta = cos_theta.masked_fill(~mask, 0) if causal else cos_theta

        s_i_expand = s_i.permute(0, 2, 3, 1).expand(-1, -1, L, -1)
        if causal:
            s_i_expand_exp = s_i_expand.masked_fill(~mask, float("-inf"))
        else:
            s_i_expand_exp = s_i_expand
        s_i_expand_exp = F.softmax(s_i_expand_exp, dim=-1)
        s_i_expand_tril = s_i_expand_exp.masked_fill(~mask, 0) if causal else s_i_expand
        s_i_expand_tril_sum = s_i_expand_tril.sum(dim=-1, keepdim=True)

        factor_res = s_i_expand_exp
        factor_zero_order = (1 / t)
        factor_first_order = (s_i_expand_tril / t) - (s_i_expand_tril_sum / (t ** 2))
        factor_remaining_orders = factor_res - factor_zero_order - factor_first_order

        if is_first_layer:
            r_t_i = (
                torch.sigmoid(gate[:, :, :, [0]]).transpose(1, 2) * factor_first_order
                + torch.sigmoid(gate[:, :, :, [1]]).transpose(1, 2) * factor_remaining_orders
                + torch.tanh(gate[:, :, :, [2]]).transpose(1, 2) * factor_zero_order
            )
        else:
            r_t_i = (
                torch.sigmoid(gate[:, :, :, [0]]).transpose(1, 2) * factor_first_order
                + torch.sigmoid(gate[:, :, :, [1]]).transpose(1, 2) * factor_remaining_orders
            )

        weight = (r_t_i * cos_theta).masked_fill(~mask, 0) if causal else r_t_i * cos_theta
        out = torch.einsum("bhli,bihd->blhd", weight, v)

    return out


class ZeroSAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        self.D = config.n_embd
        self.n_head = config.n_head
        self.d = self.D // self.n_head

        self.q = nn.Linear(self.D, self.D, bias=config.bias)
        self.k = nn.Linear(self.D, self.D, bias=config.bias)
        self.v = nn.Linear(self.D, self.D, bias=config.bias)
        self.u = nn.Linear(self.D, self.D, bias=config.bias)
        self.gate = nn.Linear(self.D, 3 * self.n_head, bias=config.bias)
        self.out_proj = nn.Linear(self.D, self.D, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        cos, sin = get_rotary_embedding(config.block_size, self.d)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", bias.view(1, 1, config.block_size, config.block_size).bool())

        self.norm = nn.LayerNorm(self.d, eps=1e-5, elementwise_affine=False)

        self.prior_mu = nn.Parameter(torch.zeros(1, 1, self.n_head, self.d))
        self.prior_log_tau = nn.Parameter(torch.zeros(1, 1, self.n_head, 1))

        self.is_first_layer = getattr(config, "is_first_layer", False)
        self.is_causal = getattr(config, "is_causal", True)
        self.init_params = getattr(config, "init_params", False)
        self.use_norm = getattr(config, "use_norm", True)
        self.use_associative = getattr(config, "use_associative", True)

        if self.init_params:
            self.apply(self._init_weights)
            self.n_layers = getattr(config, "init_n_layers", 12)
            for pn, p in self.named_parameters():
                if pn.endswith("out_proj.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, X):
        B, L, D = X.shape

        q = self.q(X).view(B, L, self.n_head, self.d)
        k = self.k(X).view(B, L, self.n_head, self.d)
        v = self.v(X).view(B, L, self.n_head, self.d)
        u = self.u(X).view(B, L, self.n_head, self.d)

        gate = self.gate(X).view(B, L, self.n_head, 3)

        s_i = self.calculate_logit(u, logit_type="deviation")

        q = apply_rotary_pos_emb(q, self.cos[:, :L], self.sin[:, :L])
        k = apply_rotary_pos_emb(k, self.cos[:, :L], self.sin[:, :L])

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        out = compute_o(
            q,
            s_i,
            k,
            v,
            gate=gate,
            causal=self.is_causal,
            mask=self.bias[:, :, 0:L, 0:L],
            associative=self.use_associative,
            is_first_layer=self.is_first_layer,
        )

        if self.use_norm:
            out = self.norm(out)

        out = out.reshape(B, L, D)
        out = self.dropout(self.out_proj(out))
        return out

    def calculate_logit(self, u, logit_type="deviation"):
        _, L, _, _ = u.shape
        t = torch.arange(1, L + 1, device=u.device, dtype=u.dtype).view(1, L, 1, 1)
        if logit_type == "deviation":
            log_tau = torch.exp(self.prior_log_tau.clip(-50, 30))
            u_sum = u.cumsum(dim=1) if self.is_causal else u.sum(dim=1, keepdim=True)
            bar_u_i = (log_tau * self.prior_mu + u_sum) / (log_tau + t)
            s_i = -(u * bar_u_i).sum(dim=-1, keepdim=True) / math.sqrt(self.d)
        elif logit_type == "quadratic":
            s_i = (u * u).sum(dim=-1, keepdim=True) / self.d
        elif logit_type == "mean":
            s_i = u.sum(dim=-1, keepdim=True) / self.d
        elif logit_type == "distance_1":
            u_mean = u.cumsum(dim=1) / t
            s_i = torch.abs(u - u_mean).sum(dim=-1, keepdim=True) / self.d
        elif logit_type == "distance_rms":
            u_mean = u.cumsum(dim=1) / t
            s_i = torch.sqrt(torch.square(u - u_mean).sum(dim=-1, keepdim=True) / self.d + 1e-8)
        else:
            raise ValueError(f"Unknown logit_type: {logit_type}")
        return s_i

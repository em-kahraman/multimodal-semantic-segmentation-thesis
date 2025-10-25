"""Self-attention variants for the transformer blocks of the hierarchical text decoder.

All modules take and return feature maps of shape (B, C, H, W). Select one with
`build_attention(name, dim, num_heads, **kwargs)`.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention over all H*W positions (torch.nn.MultiheadAttention)."""

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        out, _ = self.multihead_attn(query=x, key=x, value=x)
        return rearrange(out, "b (h w) c -> b c h w", h=h, w=w)


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention from Restormer (Zamir et al., CVPR 2022), as used in Text-IF.

    Attention is computed across channels (a C x C map per head), so the cost is linear in H*W.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = rearrange(attn @ v, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class LinformerSelfAttention(nn.Module):
    """Linformer self-attention (Wang et al., 2020), adapted from https://github.com/lucidrains/linformer (MIT).

    Keys and values are projected from sequence length `seq_len` down to `k` with learned matrices.
    Shorter inputs use the first n rows of the projections, so `seq_len` is an upper bound.
    """

    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.0):
        super().__init__()
        assert (dim % heads) == 0, "dimension must be divisible by the number of heads"
        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        dim_head = dim_head if dim_head is not None else dim // heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(self._init_proj(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(self._init_proj(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    @staticmethod
    def _init_proj(tensor):
        std = 1 / math.sqrt(tensor.shape[-1])
        return tensor.uniform_(-std, std)

    def forward(self, x):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k
        assert n <= self.seq_len, f"the sequence length of the key / values must be {self.seq_len} - {n} given"

        queries = self.to_q(x)
        keys = self.to_k(x)
        values = self.to_v(x) if not self.share_kv else keys
        proj_k = self.proj_k[:n]
        proj_v = (self.proj_v if not self.share_kv else self.proj_k)[:n]
        keys = torch.einsum("bnd,nk->bkd", keys, proj_k)
        values = torch.einsum("bnd,nk->bkd", values, proj_v)

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)
        keys, values = (t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1) for t in (keys, values))

        attn = (torch.einsum("bhnd,bhkd->bhnk", queries, keys) * (d_h**-0.5)).softmax(dim=-1)
        out = torch.einsum("bhnk,bhkd->bhnd", self.dropout(attn), values)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class LinformerAttention(nn.Module):
    """Linformer self-attention on (B, C, H, W) feature maps."""

    def __init__(self, dim, num_heads, seq_len, k=256, dropout=0.0):
        super().__init__()
        self.attn = LinformerSelfAttention(dim=dim, seq_len=seq_len, k=k, heads=num_heads, dropout=dropout)

    def forward(self, x):
        h, w = x.shape[-2:]
        out = self.attn(rearrange(x, "b c h w -> b (h w) c"))
        return rearrange(out, "b (h w) c -> b c h w", h=h, w=w)


class PerformerAttention(nn.Module):
    """Performer attention (Choromanski et al., ICLR 2021) with FAVOR+ positive random features.

    The softmax kernel is approximated with `num_random_features` orthogonal random features
    (one fixed draw, stored as a buffer so it is saved with the weights), giving linear cost in H*W.
    """

    def __init__(self, dim, num_heads, num_random_features=128, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_random_features = num_random_features
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.eps = 1e-6
        self.register_buffer("proj_matrix", self._orthogonal_random_matrix())

    def _orthogonal_random_matrix(self):
        num_blocks, remaining_rows = divmod(self.num_random_features, self.head_dim)
        blocks = []
        for _ in range(num_blocks):
            q, _ = torch.linalg.qr(torch.randn(self.head_dim, self.head_dim))
            blocks.append(q.T)
        if remaining_rows > 0:
            q, _ = torch.linalg.qr(torch.randn(self.head_dim, self.head_dim))
            blocks.append(q.T[:remaining_rows, :])
        return torch.cat(blocks, dim=0) * math.sqrt(float(self.head_dim))

    def _feature_map(self, data, is_query):
        data = (1.0 / (data.shape[-1] ** 0.25)) * data
        ratio = 1.0 / math.sqrt(self.num_random_features)
        data_dash = torch.einsum("b h n d, m d -> b h n m", data, self.proj_matrix)
        diag_data = (torch.sum(data**2, dim=-1) / 2.0).unsqueeze(-1)
        # Subtract the max before exp for numerical stability (per row for queries, global for keys)
        max_val = torch.max(data_dash, dim=-1, keepdim=True).values
        if not is_query:
            max_val = torch.max(max_val, dim=-2, keepdim=True).values
        return ratio * (torch.exp(data_dash - diag_data - max_val) + self.eps)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h d) H W -> b h (H W) d", h=self.num_heads) for t in (q, k, v))

        q_prime = self._feature_map(q, is_query=True)
        k_prime = self._feature_map(k, is_query=False)
        kv = torch.einsum("b h n m, b h n d -> b h m d", k_prime, v)
        d_inv = 1.0 / (torch.einsum("b h n m, b h m -> b h n", q_prime, k_prime.sum(dim=-2)) + self.eps)
        out = torch.einsum("b h n m, b h m d -> b h n d", q_prime, kv) * d_inv.unsqueeze(-1)

        out = rearrange(out, "b h (H W) d -> b (h d) H W", H=h, W=w)
        return self.proj_out(out)


class EfficientAttention(nn.Module):
    """Efficient Attention (Shen et al., WACV 2021), softmax normalisation variant.

    Computes softmax_k(K)^T V first, so the cost is linear in H*W. Unlike the paper, this
    implementation uses a single head (`num_heads` is unused). It also replaces the
    transformer block's identity residual with a learned 1x1 conv shortcut on the block's
    (normalised) input, which is why `replaces_residual` is set.
    """

    replaces_residual = True

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.key_channels = dim
        self.value_channels = dim
        self.queries = nn.Conv2d(dim, self.key_channels, kernel_size=1, bias=bias)
        self.keys = nn.Conv2d(dim, self.key_channels, kernel_size=1, bias=bias)
        self.values = nn.Conv2d(dim, self.value_channels, kernel_size=1, bias=bias)
        self.reprojection = nn.Conv2d(self.value_channels, dim, kernel_size=1, bias=bias)
        self.shortcut = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, _, h, w = x.shape
        shortcut = self.shortcut(x)
        queries = self.queries(x).reshape(b, self.key_channels, h * w).softmax(dim=1)
        keys = self.keys(x).reshape(b, self.key_channels, h * w).softmax(dim=2)
        values = self.values(x).reshape(b, self.value_channels, h * w)

        context = torch.bmm(keys, values.transpose(1, 2))  # (B, Ck, Cv)
        out = torch.bmm(context.transpose(1, 2), queries)  # (B, Cv, N)
        out = self.reprojection(out.reshape(b, self.value_channels, h, w))
        return out + shortcut


ATTENTION = {
    "mha": MultiHeadAttention,
    "mdta": MDTA,
    "linformer": LinformerAttention,
    "performer": PerformerAttention,
    "efficient": EfficientAttention,
}


def build_attention(name, dim, num_heads, **kwargs):
    if name not in ATTENTION:
        raise ValueError(f"Unknown attention '{name}', choose from {sorted(ATTENTION)}")
    return ATTENTION[name](dim, num_heads, **kwargs)

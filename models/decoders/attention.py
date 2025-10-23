"""Self-attention variants for the transformer blocks of the hierarchical text decoder.

All modules take and return feature maps of shape (B, C, H, W). Select one with
`build_attention(name, dim, num_heads, **kwargs)`.
"""
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


ATTENTION = {
    "mha": MultiHeadAttention,
    "mdta": MDTA,
}


def build_attention(name, dim, num_heads, **kwargs):
    if name not in ATTENTION:
        raise ValueError(f"Unknown attention '{name}', choose from {sorted(ATTENTION)}")
    return ATTENTION[name](dim, num_heads, **kwargs)

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import build_attention
from .decode_head import BaseDecodeHead
from .layers import ConvModule


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Layer normalisation over the channels of a (B, C, H, W) feature map."""

    def __init__(self, dim):
        super().__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.body(rearrange(x, "b c h w -> b (h w) c"))
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class FeedForward(nn.Module):
    """Feed-forward network with a depthwise conv (Restormer's GDFN without the gating)."""

    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        return self.project_out(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, dim, attn, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = attn
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        out = self.attn(self.norm1(x))
        # Some attention variants bring their own (learned) shortcut instead of the identity residual
        x = out if getattr(self.attn, "replaces_residual", False) else x + out
        x = x + self.ffn(self.norm2(x))
        return x


class TextGuidanceModule(nn.Module):
    """Semantic Interaction Guidance Module (SIGM), following Text-IF.

    An MLP maps the text embedding to per-channel scale and shift parameters
    (gamma, beta) that modulate the visual features: (1 + gamma) * x + beta.
    """

    def __init__(self, text_embed_dim, visual_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_embed_dim, visual_channels),
            nn.ReLU(),
            nn.Linear(visual_channels, 2 * visual_channels),
        )

    def forward(self, visual_features, text_embedding):
        gamma, beta = torch.chunk(self.mlp(text_embedding), 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * visual_features + beta


class HierarchicalTextDecoder(BaseDecodeHead):
    """Hierarchical text-guided decoder inspired by Text-IF.

    Encoder features from stages 4, 3 and 2 are decoded from coarse to fine. At each level the
    features are modulated by the text embedding (SIGM), refined by transformer blocks, upsampled
    and fused with the next encoder stage through a skip connection.

    Args:
        attention (str): Self-attention used in the transformer blocks, see `attention.ATTENTION`.
        text_embed_dim (int|None): Size of the text embedding. None disables text guidance
            (the SIGM modules are not created), which gives the no-text ablation.
        num_transformer_blocks (int): Transformer blocks per decoder level.
        attention_kwargs (dict|None): Extra arguments for the attention module.
        seq_lens (list[int]|None): Maximum sequence lengths (H*W) of the stage 2, 3 and 4 features.
            Only used, and required, by Linformer.
    """

    def __init__(
        self,
        attention="mha",
        text_embed_dim=None,
        num_transformer_blocks=2,
        attention_kwargs=None,
        seq_lens=None,
        **kwargs,
    ):
        super().__init__(input_transform="multiple_select", **kwargs)
        self.use_text = text_embed_dim is not None
        if attention == "linformer" and (seq_lens is None or len(seq_lens) != 3):
            raise ValueError("Linformer needs `seq_lens` with the sequence lengths of stages 2, 3 and 4.")
        attention_kwargs = attention_kwargs or {}
        s2_channels, s3_channels, s4_channels = self.in_channels

        def blocks(dim, num_heads, level, n):
            layers = []
            for _ in range(n):
                extra = dict(seq_len=seq_lens[level]) if attention == "linformer" else {}
                attn = build_attention(attention, dim, num_heads, **attention_kwargs, **extra)
                layers.append(TransformerBlock(dim, attn))
            return nn.Sequential(*layers)

        def guidance(channels):
            return TextGuidanceModule(text_embed_dim, channels) if self.use_text else None

        # Level 4
        self.prompt_guidance_4 = guidance(s4_channels)
        self.decoder_level4 = blocks(s4_channels, 8, 2, num_transformer_blocks)

        # Level 3
        self.up4_3 = nn.ConvTranspose2d(s4_channels, s3_channels, kernel_size=4, stride=2, padding=1)
        self.prompt_guidance_3 = guidance(s3_channels)
        self.reduce_chan_level3 = nn.Conv2d(s3_channels * 2, s3_channels, kernel_size=1, bias=False)
        self.decoder_level3 = blocks(s3_channels, 4, 1, num_transformer_blocks)

        # Level 2
        self.up3_2 = nn.ConvTranspose2d(s3_channels, s2_channels, kernel_size=4, stride=2, padding=1)
        self.prompt_guidance_2 = guidance(s2_channels)
        self.reduce_chan_level2 = nn.Conv2d(s2_channels * 2, s2_channels, kernel_size=1, bias=False)
        self.decoder_level2 = blocks(s2_channels, 2, 0, num_transformer_blocks)

        # Final refinement and output
        self.refinement = blocks(s2_channels, 2, 0, 1)[0]
        self.output_conv = ConvModule(s2_channels, self.channels, 1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def _guide(self, module, features, text_embedding):
        return module(features, text_embedding) if self.use_text else features

    def forward(self, inputs, text_embedding=None):
        if self.use_text and text_embedding is None:
            raise ValueError("This decoder is text-guided, but no text embedding was given.")
        features_s2, features_s3, features_s4 = self._transform_inputs(inputs)

        decoded_s4 = self.decoder_level4(self._guide(self.prompt_guidance_4, features_s4, text_embedding))

        upsampled_3 = self._guide(self.prompt_guidance_3, self.up4_3(decoded_s4), text_embedding)
        decoded_s3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([upsampled_3, features_s3], dim=1)))

        upsampled_2 = self._guide(self.prompt_guidance_2, self.up3_2(decoded_s3), text_embedding)
        decoded_s2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([upsampled_2, features_s2], dim=1)))

        output = self.output_conv(self.refinement(decoded_s2))
        # Upsampling to the input resolution happens in EncoderDecoder
        return self.cls_seg(output)

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from einops import rearrange
import numbers

from .decode_head import BaseDecodeHead

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
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
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            raise NotImplementedError
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class TextGuidanceModule(nn.Module):
    """
    Generates and applies text-based modulation (gamma and beta).
    """
    def __init__(self, text_embed_dim, visual_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_embed_dim, visual_channels),
            nn.ReLU(),
            nn.Linear(visual_channels, 2 * visual_channels)
        )

    def forward(self, visual_features, text_embedding):
        mod_params = self.mlp(text_embedding)
        gamma, beta = torch.chunk(mod_params, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * visual_features + beta

class HierarchicalTextDecoder(BaseDecodeHead):
    """
    Hierarchical, text-guided decoder inspired by Text-IF.
    """
    def __init__(self, num_transformer_blocks=2, **kwargs):
        super(HierarchicalTextDecoder, self).__init__(input_transform="multiple_select", **kwargs)
        
        self.text_embed_dim = 512 # CLIP's embedding dimension
        
        # DFormerv2-S channel dimensions for stages 2, 3, 4
        s2_channels, s3_channels, s4_channels = self.in_channels 
        
        # Decoder for Level 4
        self.prompt_guidance_4 = TextGuidanceModule(self.text_embed_dim, s4_channels)
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=s4_channels, num_heads=8) for _ in range(num_transformer_blocks)
        ])
        
        # Decoder for Level 3
        self.up4_3_new = nn.ConvTranspose2d(s4_channels, s3_channels, kernel_size=4, stride=2, padding=1)
        self.prompt_guidance_3 = TextGuidanceModule(self.text_embed_dim, s3_channels)
        self.reduce_chan_level3 = nn.Conv2d(s3_channels * 2, s3_channels, kernel_size=1, bias=False)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=s3_channels, num_heads=4) for _ in range(num_transformer_blocks)
        ])

        # Decoder for Level 2
        self.up3_2_new = nn.ConvTranspose2d(s3_channels, s2_channels, kernel_size=4, stride=2, padding=1)
        self.prompt_guidance_2 = TextGuidanceModule(self.text_embed_dim, s2_channels)
        self.reduce_chan_level2 = nn.Conv2d(s2_channels * 2, s2_channels, kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=s2_channels, num_heads=2) for _ in range(num_transformer_blocks)
        ])

        # Final Refinement and Output
        self.refinement = TransformerBlock(dim=s2_channels, num_heads=2)
        self.output_conv = ConvModule(
            s2_channels, self.channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )


    def forward(self, inputs, text_embedding):
        # Unpack features from the DFormerv2 encoder backbone
        features_s2, features_s3, features_s4 = self._transform_inputs(inputs)

        # Process Level 4
        modulated_s4 = self.prompt_guidance_4(features_s4, text_embedding)
        decoded_s4 = self.decoder_level4(modulated_s4)

        # Process Level 3
        upsampled_from_s4 = self.up4_3_new(decoded_s4)
        modulated_upsampled_3 = self.prompt_guidance_3(upsampled_from_s4, text_embedding)
        # Skip connection from encoder's stage 3 features
        concatenated_s3 = torch.cat([modulated_upsampled_3, features_s3], dim=1)
        reduced_s3 = self.reduce_chan_level3(concatenated_s3)
        decoded_s3 = self.decoder_level3(reduced_s3)

        # Process Level 2
        upsampled_from_s3 = self.up3_2_new(decoded_s3)
        modulated_upsampled_2 = self.prompt_guidance_2(upsampled_from_s3, text_embedding)
        # Skip connection from encoder's stage 2 features
        concatenated_s2 = torch.cat([modulated_upsampled_2, features_s2], dim=1)
        reduced_s2 = self.reduce_chan_level2(concatenated_s2)
        decoded_s2 = self.decoder_level2(reduced_s2)
        
        # Final output generation
        refined_features = self.refinement(decoded_s2)
        output = self.output_conv(refined_features)

        # The final upsampling to the original image size is handled outside this head
        # pass it to the classification layer
        output = self.cls_seg(output)
        
        return output

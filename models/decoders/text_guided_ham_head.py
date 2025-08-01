import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.ops import resize

from .decode_head import BaseDecodeHead
from .ham_head import Hamburger

class TextGuidanceModule(nn.Module):
    """
    This module contains the MLP to generate gamma and beta from a text
    embedding and then applies them to the visual features.
    """
    def __init__(self, text_embed_dim, visual_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_embed_dim, visual_channels),
            nn.ReLU(),
            nn.Linear(visual_channels, 2 * visual_channels)
        )

    def forward(self, visual_features, text_embedding):
        # Generate gamma and beta from the text embedding
        mod_params = self.mlp(text_embedding)
        gamma, beta = torch.chunk(mod_params, 2, dim=1)

        # Apply the modulation
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        modulated_features = (1 + gamma) * visual_features + beta
        return modulated_features

class TextGuidedHamHead(BaseDecodeHead):
    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super(TextGuidedHamHead, self).__init__(input_transform="multiple_select", **kwargs)
        self.ham_channels = ham_channels
        self.text_embed_dim = 512 # CLIP's embedding dimension

        # Modules from original LightHamHead
        self.squeeze = ConvModule(
            sum(self.in_channels), self.ham_channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
        )
        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)
        self.align = ConvModule(
            self.ham_channels, self.channels, 1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )

        # Create a fusion module for each feature map from the backbone
        # self.in_channels is a list of channel dimensions, e.g. [128, 256, 512]
        self.fusion_modules = nn.ModuleDict()
        for i, in_channel in enumerate(self.in_channels):
             self.fusion_modules[f'stage_{i+2}'] = TextGuidanceModule(self.text_embed_dim, in_channel)

    def forward(self, inputs, text_embedding): # MODIFIED Signature
        """Forward function."""
        inputs = self._transform_inputs(inputs) # This is a list of tensors
        
        # Modulate each feature map from the backbone individually
        modulated_f2 = self.fusion_modules['stage_2'](inputs[0], text_embedding)
        modulated_f3 = self.fusion_modules['stage_3'](inputs[1], text_embedding)
        modulated_f4 = self.fusion_modules['stage_4'](inputs[2], text_embedding)
        
        modulated_inputs = [modulated_f2, modulated_f3, modulated_f4]
        
        # Resize and concatenate the modulated features
        resized_inputs = [
            resize(level, size=modulated_inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
            for level in modulated_inputs
        ]
        x = torch.cat(resized_inputs, dim=1)
        
        x = self.squeeze(x)
        x = self.hamburger(x)
        output = self.align(x)
        output = self.cls_seg(output)
        return output

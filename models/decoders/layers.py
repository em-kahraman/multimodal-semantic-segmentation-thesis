import torch.nn as nn

NORM_LAYERS = {"BN": nn.BatchNorm2d, "SyncBN": nn.SyncBatchNorm}


class ConvModule(nn.Module):
    """Conv2d -> norm -> ReLU block.

    A minimal stand-in for `mmcv.cnn.ConvModule` that keeps its parameter names
    (`conv`, `bn`), so checkpoints trained with the mmcv version load unchanged.
    As in mmcv, the convolution only has a bias when there is no norm layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, norm_cfg=None, act_cfg=dict(type="ReLU")):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=norm_cfg is None)
        if norm_cfg is not None:
            norm_cfg = dict(norm_cfg)
            norm_layer = NORM_LAYERS[norm_cfg.pop("type")]
            requires_grad = norm_cfg.pop("requires_grad", True)
            self.bn = norm_layer(out_channels, **norm_cfg)
            for param in self.bn.parameters():
                param.requires_grad = requires_grad
        else:
            self.bn = None
        if act_cfg is not None:
            assert act_cfg["type"] == "ReLU", f"Unsupported activation {act_cfg['type']}"
            self.activate = nn.ReLU(inplace=True)
        else:
            self.activate = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x

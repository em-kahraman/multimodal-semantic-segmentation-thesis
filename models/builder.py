import torch.nn as nn
import torch.nn.functional as F

from utils.engine.logger import get_logger
from utils.init_func import init_weight

logger = get_logger()

# Channel dimensions of the four encoder stages
BACKBONE_CHANNELS = {
    "DFormerv2_S": [64, 128, 256, 512],
    "DFormerv2_B": [80, 160, 320, 512],
    "DFormerv2_L": [112, 224, 448, 640],
}


def build_text_encoder(name):
    if name == "CLIP":
        from .text_encoders.clip_encoder import ClipTextEncoder

        return ClipTextEncoder()
    if name == "JinaCLIPv2":
        from .text_encoders.jinaclipv2_encoder import JinaClipv2TextEncoder

        return JinaClipv2TextEncoder()
    raise NotImplementedError(f"Text encoder {name} not implemented.")


class EncoderDecoder(nn.Module):
    """DFormerv2 RGB-D encoder + (optionally text-guided) decoder.

    Text guidance is enabled by setting `cfg.text_encoder`. The frozen text encoder turns the
    image captions into embeddings that the decoder uses to modulate the visual features.
    """

    def __init__(
        self,
        cfg=None,
        criterion=nn.CrossEntropyLoss(reduction="none", ignore_index=255),
        norm_layer=nn.BatchNorm2d,
        syncbn=False,
    ):
        super().__init__()
        self.norm_layer = norm_layer
        self.cfg = cfg

        if cfg.backbone not in BACKBONE_CHANNELS:
            raise NotImplementedError(f"Backbone {cfg.backbone} not implemented.")
        from .encoders import DFormerv2

        backbone = getattr(DFormerv2, cfg.backbone)
        self.channels = BACKBONE_CHANNELS[cfg.backbone]

        norm_cfg = dict(type="SyncBN" if syncbn else "BN", requires_grad=True)
        drop_path_rate = cfg.drop_path_rate if cfg.drop_path_rate is not None else 0.1
        self.backbone = backbone(drop_path_rate=drop_path_rate, norm_cfg=norm_cfg)

        text_encoder_name = getattr(cfg, "text_encoder", None)
        if text_encoder_name is not None:
            logger.info(f"Using {text_encoder_name} text encoder")
            self.text_encoder = build_text_encoder(text_encoder_name)
            text_embed_dim = self.text_encoder.embed_dim
        else:
            self.text_encoder = None
            text_embed_dim = None

        head_kwargs = dict(
            in_channels=self.channels[1:],
            num_classes=cfg.num_classes,
            in_index=[1, 2, 3],
            norm_cfg=norm_cfg,
            channels=cfg.decoder_embed_dim,
        )
        if cfg.decoder == "ham":
            if self.text_encoder is not None:
                raise ValueError("The ham decoder does not support text guidance.")
            logger.info("Using Ham Decoder")
            from .decoders.ham_head import LightHamHead

            self.decode_head = LightHamHead(**head_kwargs)
        elif cfg.decoder == "HierarchicalTextDecoder":
            logger.info(
                f"Using Hierarchical Decoder with {cfg.attention} attention, "
                + ("text guidance" if text_embed_dim else "no text guidance")
            )
            from .decoders.hierarchical_text_decoder import HierarchicalTextDecoder

            self.decode_head = HierarchicalTextDecoder(
                attention=cfg.attention,
                text_embed_dim=text_embed_dim,
                seq_lens=getattr(cfg, "linformer_seq_lens", None),
                **head_kwargs,
            )
        else:
            raise NotImplementedError(f"Decoder {cfg.decoder} not implemented.")

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

    @property
    def uses_text(self):
        return self.text_encoder is not None

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info("Loading pretrained model: {}".format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info("Initing weights ...")
        init_weight(
            self.decode_head,
            nn.init.kaiming_normal_,
            self.norm_layer,
            cfg.bn_eps,
            cfg.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )

    def encode_decode(self, rgb, modal_x, text_descriptions=None):
        """Encode images with the backbone and decode them into a segmentation map of the input size."""
        x = self.backbone(rgb, modal_x)
        if len(x) == 2:  # if output is (rgb, depth) only use rgb
            x = x[0]

        if self.uses_text:
            if text_descriptions is None:
                raise ValueError("This model is text-guided, but no text descriptions were given.")
            out = self.decode_head(x, self.text_encoder(text_descriptions))
        else:
            out = self.decode_head(x)

        return F.interpolate(out, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, rgb, modal_x=None, label=None, text_descriptions=None):
        out = self.encode_decode(rgb, modal_x, text_descriptions=text_descriptions)
        if label is not None:
            return self.criterion(out, label.long())[label.long() != self.cfg.background].mean()
        return out

"""Settings shared by all hierarchical-decoder experiments on NYUDepthv2 (thesis setup).

Each experiment config imports this module, overrides what differs (attention,
text encoder, captions) and calls `setup_paths`.
"""
from .._base_ import setup_paths
from .._base_.datasets.NYUDepthv2 import *

"""Settings for network"""
C.backbone = "DFormerv2_S"
C.pretrained_model = "checkpoints/pretrained/DFormerv2_Small_pretrained.pth"
C.decoder = "HierarchicalTextDecoder"
C.attention = "mha"
C.decoder_embed_dim = 512
C.text_encoder = "JinaCLIPv2"
C.text_annotation_file = "captions/NYUDepthv2/llava-next.json"
C.optimizer = "AdamW"
C.seed = 54321

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 16
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.25
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [0.5, 0.75, 1, 1.25, 1.5]  # with --mst, train.py and eval.py use these scales (hard-coded there)
C.eval_flip = True
C.eval_crop_size = [480, 640]  # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 200
C.checkpoint_step = 25

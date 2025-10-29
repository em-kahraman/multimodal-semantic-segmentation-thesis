"""Main model: multi-head self-attention decoder, JinaCLIP v2 text encoder, LLaVA-1.6 captions."""
from ._hierarchical import *

# The text-guided SUN RGB-D model was trained for 200 epochs (the ablation for 300)
C.nepochs = 200
C.checkpoint_start_epoch = 130

setup_paths(C, "DFormerv2_S_text")

"""Decoder with Efficient Attention (Shen et al.)."""
from ._hierarchical import *

C.attention = "efficient"

setup_paths(C, "DFormerv2_S_text_efficient")

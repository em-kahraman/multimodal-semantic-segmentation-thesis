"""Decoder with Multi-Dconv Head Transposed Attention (Restormer)."""
from ._hierarchical import *

C.attention = "mdta"

setup_paths(C, "DFormerv2_S_text_mdta")

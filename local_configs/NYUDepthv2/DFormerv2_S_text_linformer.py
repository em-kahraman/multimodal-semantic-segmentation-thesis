"""Decoder with Linformer attention."""
from ._hierarchical import *

C.attention = "linformer"
# Upper bounds of H*W for the stage 2, 3 and 4 features: the largest validation scale
# (1.5x of 480x640, padded to 736x960). These set the shape of the learned projections.
C.linformer_seq_lens = [11040, 2760, 690]

setup_paths(C, "DFormerv2_S_text_linformer")

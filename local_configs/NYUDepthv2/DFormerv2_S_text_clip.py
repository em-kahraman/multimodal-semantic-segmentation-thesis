"""Main model with the CLIP text encoder (captions truncated to 77 tokens)."""
from ._hierarchical import *

C.text_encoder = "CLIP"

setup_paths(C, "DFormerv2_S_text_clip")

"""Decoder with Performer (FAVOR+) attention."""
from ._hierarchical import *

C.attention = "performer"

setup_paths(C, "DFormerv2_S_text_performer")

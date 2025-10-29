"""Ablation: the same hierarchical decoder without text guidance."""
from ._hierarchical import *

C.text_encoder = None
C.text_annotation_file = None

setup_paths(C, "DFormerv2_S_no_text")

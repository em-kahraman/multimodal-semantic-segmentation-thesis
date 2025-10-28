"""Main model with InternVL3-38B captions."""
from ._hierarchical import *

C.text_annotation_file = "captions/NYUDepthv2/internvl3.json"

setup_paths(C, "DFormerv2_S_text_internvl3")

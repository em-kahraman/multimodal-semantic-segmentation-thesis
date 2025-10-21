import torch
import torch.nn as nn
from transformers import AutoModel

# Pinned to the revisions used for the thesis experiments (the Hub repos have changed since).
# The text tower's own code (jinaai/xlm-roberta-flash-implementation) is always fetched at its
# latest revision by JinaCLIP; its later fixes change the embeddings only at fp16 rounding level.
MODEL_REVISION = "344d954da76eb8ad47a7aaff42d012e30c15b8fe"
CODE_REVISION = "39e6a55ae971b59bea6e44675d237c99762e7ee2"  # jinaai/jina-clip-implementation


class JinaClipv2TextEncoder(nn.Module):
    """Frozen JinaCLIP v2 text encoder (up to 8192 tokens), producing 1024-d text embeddings."""

    def __init__(self, model_name="jinaai/jina-clip-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=MODEL_REVISION,
            code_revision=CODE_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.requires_grad_(False)
        self.embed_dim = self.model.config.projection_dim

    def forward(self, text_descriptions):
        """Encode a list of strings into a (B, embed_dim) tensor."""
        # encode_text runs the model in eval mode and returns L2-normalised embeddings as a NumPy array
        embedding = self.model.encode_text(text_descriptions, show_progress_bar=False)
        device = next(self.model.parameters()).device
        return torch.from_numpy(embedding).to(device)

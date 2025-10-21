import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class ClipTextEncoder(nn.Module):
    """Frozen CLIP text encoder. Captions are truncated to CLIP's 77-token context."""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model.requires_grad_(False)
        self.embed_dim = self.text_model.config.hidden_size

    def forward(self, text_descriptions):
        """Encode a list of strings into a (B, embed_dim) tensor (pooled output of the text transformer)."""
        inputs = self.tokenizer(
            text=text_descriptions,
            padding=True,
            truncation=True,
            max_length=self.text_model.config.max_position_embeddings,
            return_tensors="pt",
        ).to(self.text_model.device)
        return self.text_model(**inputs).pooler_output

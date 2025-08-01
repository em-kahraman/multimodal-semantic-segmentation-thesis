import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

class ClipTextEncoder(nn.Module): # Renamed for clarity
    """
    Standalone module to encode text descriptions into a feature embedding.
    """
    def __init__(self):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze the CLIP text model to preserve linguistic knowledge
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False

    def forward(self, text_descriptions):
        """
        Args:
            text_descriptions (list of str): The batch of text descriptions.
        
        Returns:
            text_embedding (Tensor): The raw text feature embedding from CLIP.
        """
        # Add truncation=True to handle long captions, it will cut off text longer than the model can handle
        # Also explicitly set the max_length from the model's config for robustness
        inputs = self.tokenizer(
            text=text_descriptions,
            padding=True,
            truncation=True,
            max_length=self.clip_model.config.text_config.max_position_embeddings, # usually 77 for CLIP
            return_tensors="pt"
        ).to(self.clip_model.device)
        
        text_outputs = self.clip_model.text_model(**inputs)
        text_embedding = text_outputs.pooler_output
        
        return text_embedding

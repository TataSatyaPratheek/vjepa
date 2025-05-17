import torch
import torch.nn as nn
from transformers import ViTModel
from torch.utils.checkpoint import checkpoint

class TinyVJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ViTModel.from_pretrained("google/vit-tiny-patch16-224")
        self.encoder.requires_grad_(False)  # Freeze encoder
        
        # Memory-efficient predictor
        self.predictor = nn.Sequential(
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Linear(128, 192)
        )
        
        # Enable gradient checkpointing
        self.encoder.gradient_checkpointing_enable()

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.flatten(0, 1)  # Combine batch and time
        
        # Checkpoint encoder to save memory
        features = checkpoint(self.encoder, x).last_hidden_state
        features = features.unflatten(0, (B, T))
        
        return self.predictor(features)

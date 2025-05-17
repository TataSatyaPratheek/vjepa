import torch
import torch.nn as nn
from transformers import ViTModel, AutoImageProcessor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

class TinyVJEPA(nn.Module):
    def __init__(self, pretrained="google/vit-tiny-patch16-224", freeze_encoder=True):
        super().__init__()
        # Use safetensors for faster loading and better compatibility
        self.encoder = ViTModel.from_pretrained(
            pretrained, 
            use_safetensors=True,
            torch_dtype=torch.float16  # Keep in fp16 for memory savings
        )
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained)
        
        # Freeze encoder to save memory during training
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        
        # Memory-efficient predictor with layer norm
        hidden_dim = 128
        embed_dim = self.encoder.config.hidden_size
        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Enable gradient checkpointing
        self.encoder.gradient_checkpointing_enable()
        
        # For efficient M1 inference
        self._device_setup_done = False

    def setup_for_inference(self):
        """Configure model for M1 inference"""
        if not self._device_setup_done and torch.backends.mps.is_available():
            # Use mixed precision for inference
            self.encoder = self.encoder.to(torch.float16)
            self._device_setup_done = True

    def forward(self, x, return_all_features=False):
        B, T = x.shape[:2]
        x = x.flatten(0, 1)  # Combine batch and time
        
        # Use smaller chunks for M1's limited memory
        chunk_size = min(8, x.shape[0])  # Process in chunks of max 8 frames
        
        # Checkpoint encoder to save memory
        features_list = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i+chunk_size]
            # Use the checkpoint function with use_reentrant=False for better memory efficiency
            chunk_features = checkpoint(
                lambda x: self.encoder(x).last_hidden_state, 
                chunk, 
                use_reentrant=False
            )
            features_list.append(chunk_features)
            
        features = torch.cat(features_list, dim=0)
        features = features.unflatten(0, (B, T))
        
        if return_all_features:
            return self.predictor(features), features
        
        return self.predictor(features)

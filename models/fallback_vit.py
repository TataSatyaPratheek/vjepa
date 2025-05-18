# Create new file models/fallback_vit.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger("fallback_vit")

class FallbackViT(nn.Module):
    """A minimal ViT model for when the pretrained model can't be loaded"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        logger.info("Initializing minimal ViT fallback model")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.config = type('obj', (object,), {
            'hidden_size': embed_dim
        })
        
        # Basic patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Basic transformer
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*2)
            for _ in range(4)  # Just 4 layers to keep it light
        ])
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Process through transformer
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        class NestedOutput:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
        
        return NestedOutput(x)
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.models.autoencoder_kl import AutoencoderKL
from torch.optim import AdamW
from typing import Tuple

class LatentDiffusionDecoder(nn.Module):
    def __init__(self, 
                 latent_dim: int = 192,
                 temporal_dim: int = 8,
                 scaling_factor: float = 0.18215):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.temporal_dim = temporal_dim
        
        # Lightweight diffusion model for video latents
        self.diffusion_model = UNet2DConditionModel(
            sample_size=(temporal_dim, 16, 16),
            in_channels=latent_dim,
            out_channels=latent_dim,
            layers_per_block=2,
            block_out_channels=(128, 256),
            attention_head_dim=8,
            norm_num_groups=16,
            time_embedding_type="positional"
        )
        
        # Initialize with pretrained VAE decoder components
        self.vae_decoder = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            subfolder="vae",
            in_channels=latent_dim,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"]*3,
            up_block_types=["UpDecoderBlock2D"]*3,
            block_out_channels=(128, 256, 512),
            latent_channels=latent_dim,
            norm_num_groups=16
        ).decoder
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            prediction_type="epsilon"
        )
        
        # EMA for training stability
        self.ema_decay = 0.9999
        self.ema_model = self._create_ema_model()

    def _create_ema_model(self):
        ema_model = self.diffusion_model
        for param in ema_model.parameters():
            param.requires_grad_(False)
        return ema_model

    def update_ema(self):
        with torch.no_grad():
            for param, ema_param in zip(self.diffusion_model.parameters(), 
                                      self.ema_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay)

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode videos to latent space with proper scaling"""
        # videos: [B, T, C, H, W]
        B, T = videos.shape[:2]
        videos = videos.flatten(0, 1)  # [B*T, C, H, W]
        
        latents = self.vae_encoder(videos).latent_dist.sample()
        return latents.unflatten(0, (B, T)) * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space with temporal upsampling"""
        # latents: [B, T, C, H, W]
        B, T = latents.shape[:2]
        latents = latents / self.scaling_factor
        
        # Temporal upsampling
        if T < self.temporal_dim:
            latents = nn.functional.interpolate(
                latents.permute(0,2,1,3,4), 
                size=(self.temporal_dim, *latents.shape[-2:]), 
                mode='nearest'
            ).permute(0,2,1,3,4)
        
        # Decode each frame
        frames = []
        for t in range(latents.size(1)):
            frame = self.vae_decoder(latents[:, t]).sample
            frames.append(frame)
        
        return torch.stack(frames, dim=1)  # [B, T, C, H, W]

    def forward(self, 
                noisy_latents: torch.Tensor,
                timesteps: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        # Add temporal attention conditioning
        return self.diffusion_model(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=context
        ).sample

    def compute_loss(self,
                     target_latents: torch.Tensor,
                     pred_latents: torch.Tensor) -> torch.Tensor:
        # Use huber loss for stability
        return torch.nn.functional.huber_loss(
            pred_latents, target_latents, reduction="mean"
        )

    @torch.no_grad()
    def generate(self,
                 num_samples: int = 1,
                 num_inference_steps: int = 50,
                 context: torch.Tensor = None) -> torch.Tensor:
        # Initialize random noise
        latents = torch.randn(
            num_samples, 
            self.temporal_dim,
            self.diffusion_model.in_channels,
            *self.diffusion_model.sample_size[1:]
        ).to(self.device)
        
        # Configure scheduler
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            # Predict noise residual
            noise_pred = self.ema_model(
                latents, 
                t.unsqueeze(0).repeat(num_samples),
                context
            )
            
            # Compute previous step
            latents = self.noise_scheduler.step(
                noise_pred, t, latents
            ).prev_sample
            
        return self.decode(latents)

    def configure_optimizers(self):
        return AdamW(
            self.diffusion_model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

    @property
    def device(self):
        return next(self.parameters()).device

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import os
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DModel
from models.vjepa_tiny import TinyVJEPA
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

class LatentDiffuser(nn.Module):
    def __init__(self, latent_dim: int = 192, device=None) -> None:
        super(LatentDiffuser, self).__init__()
        # Set device - prioritize MPS for M1 Macs
        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available() 
            else torch.device("cpu")
        )
        
        # Create a smaller UNet for M1 Macs
        self.model = UNet2DModel(
            sample_size=(16, 16),  # Match ViT patch grid
            in_channels=latent_dim,
            out_channels=latent_dim,
            layers_per_block=1,  # Reduce from 2 to 1 to save memory
            block_out_channels=(64, 128),  # Keep small channels for M1
            norm_num_groups=8,
            use_memory_efficient_attention=True,  # Important for M1 memory savings
        ).to(self.device)
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        # For efficient M1 inference, use fp16
        if torch.backends.mps.is_available():
            self.model = self.model.to(torch.float16)
        
        print(f"LatentDiffuser initialized on device: {self.device}")

    def _generate(self, num_samples: int = 1, progress=True) -> torch.Tensor:
        # Initialize random latent
        sample_size = self.model.config.sample_size
        if isinstance(sample_size, int):
            sample_size = (sample_size, sample_size)
            
        latent = torch.randn(
            num_samples, 
            self.model.in_channels,
            *sample_size
        ).to(self.device)
        
        # Use buffer to reduce memory transfers on M1
        if torch.backends.mps.is_available():
            latent = latent.to(torch.float16)  # Use fp16 for M1
        
        # Diffusion process
        timesteps = self.noise_scheduler.timesteps
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Diffusion")
            
        for t in timesteps:
            timestep = int(t)
            
            # Use no_grad to save memory 
            with torch.no_grad():
                # Process in chunks if needed for large batches
                if num_samples > 2 and torch.backends.mps.is_available():
                    noise_preds = []
                    for i in range(0, latent.shape[0], 2):
                        chunk = latent[i:i+2]
                        chunk_pred = self.model(chunk, timestep).sample
                        noise_preds.append(chunk_pred)
                    noise_pred = torch.cat(noise_preds, dim=0)
                else:
                    noise_pred = self.model(latent, timestep).sample
                
                # Update latent
                step_output = self.noise_scheduler.step(
                    noise_pred, timestep, latent
                )
                latent = step_output.prev_sample
                
                # Clean memory on M1
                if torch.backends.mps.is_available() and timestep % 100 == 0:
                    torch.mps.empty_cache()
            
        return latent

    @torch.no_grad()
    def generate(self, num_samples: int = 1) -> torch.Tensor:
        # Generate and move to CPU when done for display
        return self._generate(num_samples).cpu()

def generate_variations(
    encoder: TinyVJEPA,
    diffuser: LatentDiffuser,
    num_samples: int = 2,
    output_dir: str = "outputs"
) -> torch.Tensor:
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating latent samples...")
    # Generate latent samples
    latents = diffuser.generate(num_samples)
    print(f"Generated latents of shape: {latents.shape}")
    
    # Setup encoder for efficient inference
    encoder.setup_for_inference()
    device = latents.device
    
    # Move encoder to the same device
    encoder = encoder.to(device)
    if torch.backends.mps.is_available():
        encoder = encoder.to(torch.float16)
    
    with torch.no_grad():
        # Process in smaller batches for M1 memory constraints
        image_list = []
        
        for i in range(0, latents.shape[0], 1):
            batch_latents = latents[i:i+1]
            
            # Use ViT decoder more efficiently
            patches = encoder.encoder.layernorm(
                batch_latents.flatten(start_dim=1, end_dim=2)
            )
            
            # Reshape to 2D grid
            grid_size = int(patches.shape[1] ** 0.5)
            images = patches.reshape(
                batch_latents.shape[0], 
                grid_size, 
                grid_size, 
                -1
            ).permute(0, 3, 1, 2)
            
            # Upsample to full resolution using bilinear upsampling (memory efficient)
            images = torch.nn.functional.interpolate(
                images, 
                scale_factor=16, 
                mode='bilinear',
                align_corners=False
            )
            
            # Normalize for visualization
            images = images - images.min()
            images = images / images.max() if images.max() > 0 else images
            
            image_list.append(images.cpu())
            
            # Save individual images
            for j, img in enumerate(images):
                img_path = os.path.join(output_dir, f"sample_{i+j}.png")
                # Convert to numpy for matplotlib
                img_np = img.permute(1, 2, 0).numpy()
                plt.figure(figsize=(8, 8))
                plt.imshow(img_np)
                plt.axis('off')
                plt.savefig(img_path, bbox_inches='tight')
                plt.close()
        
        # Concatenate results
        images = torch.cat(image_list, dim=0)
        
    return images

def main():
    parser = argparse.ArgumentParser(description="Generate video frames using the trained model")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--encoder_path", type=str, default=None, help="Path to encoder checkpoint")
    parser.add_argument("--latent_dim", type=int, default=192, help="Latent dimension")
    args = parser.parse_args()
    
    # Initialize models
    encoder = TinyVJEPA()
    
    # Load encoder if specified
    if args.encoder_path and os.path.exists(args.encoder_path):
        state_dict = torch.load(args.encoder_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded encoder from {args.encoder_path}")
    
    # Use MPS if available, otherwise CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    diffuser = LatentDiffuser(latent_dim=args.latent_dim, device=device)
    
    # Generate variations
    images = generate_variations(
        encoder=encoder,
        diffuser=diffuser,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    print(f"Generated {args.num_samples} samples. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()

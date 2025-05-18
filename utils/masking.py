import torch
import logging

logger = logging.getLogger("masking")

def create_tube_mask(video_shape, mask_ratio=0.75):
    try:
        B, T, C, H, W = video_shape
        patch_size = 16
        num_patches = (H//patch_size) * (W//patch_size)
        
        # Generate spatial-temporal mask
        num_keep = int(num_patches * (1 - mask_ratio))
        # Use more efficient memory layout for M1
        device = video_shape[0].device if isinstance(video_shape[0], torch.Tensor) else None
        
        # Optimize by creating a single large mask and then splitting
        all_masks = torch.zeros(B * T, num_patches, device=device)
        
        for b in range(B):
            # Create consistent mask across time
            spatial_mask = torch.zeros(num_patches, device=device)
            spatial_mask[:num_keep] = 1
            spatial_mask = spatial_mask[torch.randperm(num_patches)]
            
            # Repeat across time dimension more efficiently
            all_masks[b*T:(b+1)*T] = spatial_mask.unsqueeze(0).expand(T, -1)
        
        # Reshape efficiently
        mask = all_masks.view(B, T, 1, H//patch_size, W//patch_size)
        
        # Use efficient upsampling approach
        mask = mask.repeat_interleave(patch_size, dim=3)
        mask = mask.repeat_interleave(patch_size, dim=4)
        
        return mask
    except Exception as e:
        logger.error(f"Error creating tube mask: {e}")
        # Return sensible fallback if error occurs
        return torch.ones(video_shape)  # No masking as fallback

import torch

def create_tube_mask(video_shape, mask_ratio=0.75):
    B, T, C, H, W = video_shape
    patch_size = 16
    num_patches = (H//patch_size) * (W//patch_size)
    
    # Generate spatial-temporal mask
    num_keep = int(num_patches * (1 - mask_ratio))
    # Use more efficient memory layout for M1
    mask = torch.zeros(B, T, num_patches, device=video_shape[0].device if isinstance(video_shape[0], torch.Tensor) else None)
    
    for b in range(B):
        # Create consistent mask across time
        spatial_mask = torch.zeros(num_patches)
        spatial_mask[:num_keep] = 1
        spatial_mask = spatial_mask[torch.randperm(num_patches)]
        mask[b] = spatial_mask.unsqueeze(0).repeat(T, 1)
    
    # Reshape to match video dimensions more clearly
    mask = mask.reshape(B, T, 1, H//patch_size, W//patch_size)
    # Upsample to match video resolution
    mask = mask.repeat_interleave(patch_size, dim=3)
    mask = mask.repeat_interleave(patch_size, dim=4)
    
    return mask

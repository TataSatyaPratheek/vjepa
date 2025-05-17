def create_tube_mask(video_shape, mask_ratio=0.75):
    B, T, C, H, W = video_shape
    patch_size = 16
    num_patches = (H//patch_size) * (W//patch_size)
    
    # Generate spatial-temporal mask
    num_keep = int(num_patches * (1 - mask_ratio))
    mask = torch.zeros(B, T, num_patches)
    
    for b in range(B):
        # Create consistent mask across time
        spatial_mask = torch.zeros(num_patches)
        spatial_mask[:num_keep] = 1
        spatial_mask = spatial_mask[torch.randperm(num_patches)]
        mask[b] = spatial_mask.unsqueeze(0).repeat(T, 1)
    
    return mask.reshape(B, T*num_patches)

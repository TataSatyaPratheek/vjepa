import torch
import logging
import numpy as np

logger = logging.getLogger("masking")

class MaskCollator:
    """Mask collator following official V-JEPA patterns"""
    def __init__(self, 
                 input_size=(224, 224),
                 patch_size=16,
                 enc_mask_scale=(0.2, 0.8),
                 pred_mask_scale=(0.15, 0.2),
                 aspect_ratio=(0.3, 3.0),
                 nenc=1,
                 npred=2):
        self.input_size = input_size
        self.patch_size = patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.n_patches = self.height * self.width
    
    def __call__(self):
        """Generate masks following official implementation"""
        masks_enc, masks_pred = [], []
        
        for _ in range(self.nenc):
            mask = self._generate_mask(self.enc_mask_scale)
            masks_enc.append(mask)
            
        for _ in range(self.npred):
            mask = self._generate_mask(self.pred_mask_scale)
            masks_pred.append(mask)
            
        return masks_enc, masks_pred
    
    def _generate_mask(self, mask_scale):
        """Generate a single mask with given scale"""
        scale = np.random.uniform(*mask_scale)
        aspect_ratio = np.random.uniform(*self.aspect_ratio)
        
        h = int(self.height * np.sqrt(scale / aspect_ratio))
        w = int(self.width * np.sqrt(scale * aspect_ratio))
        
        h = min(h, self.height)
        w = min(w, self.width)
        
        top = np.random.randint(0, self.height - h + 1)
        left = np.random.randint(0, self.width - w + 1)
        
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[top:top+h, left:left+w] = True
        
        return mask

def create_tube_mask(video_shape, mask_ratio=0.75):
    """Enhanced tube masking using official patterns"""
    try:
        B, T, C, H, W = video_shape
        patch_size = 16
        
        # Use official mask collator for better masking strategy
        mask_collator = MaskCollator(
            input_size=(H, W),
            patch_size=patch_size,
            enc_mask_scale=(mask_ratio-0.1, mask_ratio+0.1), # Example, adjust as needed
            pred_mask_scale=(0.1, 0.3) # Example, adjust as needed
        )
        
        # Generate masks for each batch item
        all_batch_masks = []
        for _ in range(B): # Iterate per batch item
            temporal_masks = []
            for _ in range(T): # Iterate per time step in the video
                # We are interested in encoder masks for the input video
                # The MaskCollator returns a list of encoder masks and a list of predictor masks.
                # We'll take the first encoder mask.
                enc_masks, _ = mask_collator() 
                # Assuming enc_masks[0] is a numpy array [H_patches, W_patches]
                # Convert numpy mask to torch tensor
                spatial_mask_np = enc_masks[0] 
                spatial_mask = torch.from_numpy(spatial_mask_np).float() # Ensure float for possible later ops
                temporal_masks.append(spatial_mask)
            
            # Stack masks for the time dimension for the current batch item
            # Resulting shape: [T, H_patches, W_patches]
            stacked_temporal_mask = torch.stack(temporal_masks, dim=0)
            all_batch_masks.append(stacked_temporal_mask)
        
        # Stack all batch masks together
        # Resulting shape: [B, T, H_patches, W_patches]
        mask = torch.stack(all_batch_masks, dim=0)
        
        # Expand dimensions to match video_shape for channel and patch size
        # Add channel dimension: [B, T, 1, H_patches, W_patches]
        mask = mask.unsqueeze(2) 
        
        # Upsample to full resolution
        mask = mask.repeat_interleave(patch_size, dim=3)
        mask = mask.repeat_interleave(patch_size, dim=4)
        
        return mask
    except Exception as e:
        logger.error(f"Error creating tube mask: {e}")
        return torch.ones(video_shape)  # No masking as fallback

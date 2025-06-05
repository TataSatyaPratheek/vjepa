import torch
import numpy as np
import logging

logger = logging.getLogger("mask_collators")

class TubeMaskCollator:
    """
    Tube masking collator following official V-JEPA patterns
    """
    def __init__(self, 
                 input_size=(224, 224),
                 patch_size=16,
                 mask_scale=(0.15, 0.8),
                 aspect_ratio=(0.3, 3.0),
                 nenc=1,
                 npred=4,
                 temporal_consistency=True):
        self.input_size = input_size
        self.patch_size = patch_size
        self.mask_scale = mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.temporal_consistency = temporal_consistency
        
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.n_patches = self.height * self.width
        
    def __call__(self, temporal_length=8):
        """Generate encoder and predictor masks for video sequence"""
        masks_enc = []
        masks_pred = []
        
        # Generate encoder masks (context)
        for _ in range(self.nenc):
            if self.temporal_consistency:
                mask = self._generate_tube_mask(temporal_length)
            else:
                mask = self._generate_frame_masks(temporal_length)
            masks_enc.append(mask)
            
        # Generate predictor masks (target regions to predict)
        for _ in range(self.npred):
            mask = self._generate_prediction_mask(temporal_length)
            masks_pred.append(mask)
            
        return masks_enc, masks_pred
    
    def _generate_tube_mask(self, temporal_length):
        """Generate temporal tube mask"""
        scale = np.random.uniform(*self.mask_scale)
        aspect_ratio = np.random.uniform(*self.aspect_ratio)
        
        mask_area = scale * self.n_patches
        mask_height = int(np.sqrt(mask_area / aspect_ratio))
        mask_width = int(mask_area / mask_height)
        
        mask_height = min(mask_height, self.height)
        mask_width = min(mask_width, self.width)
        
        top = np.random.randint(0, self.height - mask_height + 1)
        left = np.random.randint(0, self.width - mask_width + 1)
        
        spatial_mask = np.zeros((self.height, self.width), dtype=bool)
        spatial_mask[top:top+mask_height, left:left+mask_width] = True
        
        # Extend to temporal dimension
        tube_mask = np.tile(spatial_mask[None, :, :], (temporal_length, 1, 1))
        
        return torch.from_numpy(tube_mask)
    
    def _generate_frame_masks(self, temporal_length):
        """Generate independent masks per frame"""
        masks = []
        for t in range(temporal_length):
            scale = np.random.uniform(*self.mask_scale)
            aspect_ratio = np.random.uniform(*self.aspect_ratio)
            
            mask_area = scale * self.n_patches
            mask_height = int(np.sqrt(mask_area / aspect_ratio))
            mask_width = int(mask_area / mask_height)
            
            mask_height = min(mask_height, self.height)
            mask_width = min(mask_width, self.width)
            
            top = np.random.randint(0, self.height - mask_height + 1)
            left = np.random.randint(0, self.width - mask_width + 1)
            
            frame_mask = np.zeros((self.height, self.width), dtype=bool)
            frame_mask[top:top+mask_height, left:left+mask_width] = True
            masks.append(frame_mask)
            
        return torch.from_numpy(np.stack(masks))
    
    def _generate_prediction_mask(self, temporal_length):
        """Generate smaller masks for prediction targets"""
        pred_scale = (self.mask_scale[0] * 0.5, self.mask_scale[1] * 0.7)
        scale = np.random.uniform(*pred_scale)
        aspect_ratio = np.random.uniform(*self.aspect_ratio)
        
        mask_area = scale * self.n_patches
        mask_height = int(np.sqrt(mask_area / aspect_ratio))
        mask_width = int(mask_area / mask_height)
        
        mask_height = min(mask_height, self.height)
        mask_width = min(mask_width, self.width)
        
        top = np.random.randint(0, self.height - mask_height + 1)
        left = np.random.randint(0, self.width - mask_width + 1)
        
        spatial_mask = np.zeros((self.height, self.width), dtype=bool)
        spatial_mask[top:top+mask_height, left:left+mask_width] = True
        tube_mask = np.tile(spatial_mask[None, :, :], (temporal_length, 1, 1))
        
        return torch.from_numpy(tube_mask)

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Tuple
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from pytorchvideo.transforms import (
    ApplyTransformToKey
)
from pytorchvideo.data import labeled_video_dataset, make_clip_sampler
from pytorchvideo.transforms import UniformTemporalSubsample

class HMDB51Wrapper:
    def __init__(
        self, 
        root_path: str, 
        split: str = 'train', 
        clip_length: int = 8, 
        frame_rate: int = 5,
        frame_size: int = 112
    ):
        # Memory-efficient implementation for M1 MacBook
        self.root_path = root_path
        self.split = split
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        
        # Get class folders
        self.classes = sorted([d for d in os.listdir(root_path) 
                          if os.path.isdir(os.path.join(root_path, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Build file list
        self.video_files = []
        for class_name in self.classes:
            class_dir = os.path.join(root_path, class_name)
            if os.path.isdir(class_dir):
                files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                         if f.endswith('.mp4')]
                self.video_files.extend([(f, self.class_to_idx[class_name]) for f in files])
        
        # Efficient preprocessing transforms
        self.transform = Compose([
            UniformTemporalSubsample(self.clip_length),
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    # Memory-efficient preprocessing
                    Resize(frame_size),
                    CenterCrop(frame_size),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406]),
                ])
            )
        ])
        
        # Create sampler with small clip size for memory efficiency
        self.dataset = labeled_video_dataset(
            data_path=root_path,
            clip_sampler=make_clip_sampler(
                "random" if split == 'train' else "uniform", 
                clip_length/frame_rate
            ),
            transform=self.transform,
            decode_audio=False,  # Disable audio to save memory
        )
        
        print(f"Loaded {len(self.dataset)} video clips from {len(self.classes)} classes")
    
    def __getitem__(self, index: int):
        # Process videos efficiently for M1
        try:
            result = self.dataset[index]
            video, label, _ = result
            # Ensure video is in channel-first format (B,C,T,H,W)
            if video.shape[1] != 3:  # If not channel-first
                video = video.permute(0, 3, 1, 2)
            return video, torch.tensor(label)
        except Exception as e:
            print(f"Error loading video at index {index}: {e}")
            # Return a small zero tensor as fallback
            return torch.zeros(self.clip_length, 3, 112, 112), torch.tensor(0)
    
    def __len__(self):
        return len(self.dataset)

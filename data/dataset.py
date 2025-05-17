from pytorchvideo.data import labeled_video_dataset, make_clip_sampler  # type: ignore
from torchvision.transforms import Compose
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize
)
from torchvision.transforms import Lambda
import torch
from typing import Any
import os

def to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0

class HMDB51Wrapper:
    def __init__(self, root_path: str, split: str = 'train', clip_length: int = 8, frame_rate: int = 5):
        self.dataset: torch.utils.data.Dataset[Any] = labeled_video_dataset(
            data_path=root_path,
            clip_sampler=make_clip_sampler("random" if split == 'train' else "uniform", clip_length/frame_rate),
            transform=Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose([
                        Lambda(to_float_tensor),
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                ),
            ])
        )
        self.classes = sorted(os.listdir(root_path))
        
    def __getitem__(self, index: int):
        video, label, _ = self.dataset[index]
        return video.permute(0, 3, 1, 2), torch.tensor(label)

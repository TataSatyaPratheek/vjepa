import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Tuple
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import random
import av
import glob
import cv2
import logging
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("hmdb51")

class VideoReader:
    """Custom video reader to replace PyTorchVideo dependency"""
    def __init__(self, video_path, clip_length=8, frame_rate=None):
        self.video_path = video_path
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        
    def read_video(self):
        try:
            # Use PyAV for efficient video reading
            container = av.open(self.video_path)
            frames = []
            
            # Get video stream
            stream = container.streams.video[0]
            total_frames = stream.frames
            
            if total_frames == 0:
                # Some files don't report frames correctly, use OpenCV as fallback
                return self._read_with_opencv()
            
            # Determine sampling rate to achieve desired clip length
            if self.frame_rate is not None:
                stream.codec_context.skip_frame = "NONKEY"
                step = int(stream.average_rate / self.frame_rate)
            else:
                # Sample evenly to get clip_length frames
                step = max(1, total_frames // self.clip_length)

            # Sample frames
            for i, frame in enumerate(container.decode(stream)):
                if i % step == 0 and len(frames) < self.clip_length:
                    # Convert to numpy array
                    img = frame.to_ndarray(format='rgb24')
                    frames.append(img)
                
                if len(frames) == self.clip_length:
                    break
            
            container.close()
            
            # If we didn't get enough frames, pad with the last frame
            while len(frames) < self.clip_length and len(frames) > 0:
                frames.append(frames[-1])
                
            return frames
        except Exception as e:
            log.error(f"Error reading video {self.video_path}: {e}")
            return self._read_with_opencv()
    
    def _read_with_opencv(self):
        """Fallback to OpenCV if PyAV fails"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            frames = []
            
            if not cap.isOpened():
                log.error(f"Cannot open video: {self.video_path}")
                return []
            
            # Get total frames and FPS
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Determine sampling rate
            if self.frame_rate is not None:
                step = int(fps / self.frame_rate)
            else:
                step = max(1, total_frames // self.clip_length)
            
            frame_indices = [i*step for i in range(min(self.clip_length, total_frames))]
            
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if i in frame_indices:
                    # Convert from BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    
                if len(frames) == self.clip_length:
                    break
                    
            cap.release()
            
            # If we didn't get enough frames, pad with the last frame
            while len(frames) < self.clip_length and len(frames) > 0:
                frames.append(frames[-1])
                
            return frames
        except Exception as e:
            log.error(f"OpenCV fallback failed for {self.video_path}: {e}")
            return []

class HMDB51Wrapper:
    def __init__(
        self, 
        root_path: str, 
        split: str = 'train', 
        clip_length: int = 8, 
        frame_rate: int = 5,
        frame_size: int = 112,
        cache_mode: str = 'none'  # Options: 'none', 'memory', 'disk'
    ):
        # Setup logging
        self.logger = logging.getLogger("hmdb51")
        self.logger.info(f"Initializing HMDB51Wrapper with root_path={root_path}, split={split}")
        
        # Memory-efficient implementation for M1 MacBook
        self.root_path = root_path
        self.split = split
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.frame_size = frame_size
        self.cache_mode = cache_mode
        self.cache = {}
        
        # Get class folders
        try:
            self.classes = sorted([d for d in os.listdir(root_path) 
                              if os.path.isdir(os.path.join(root_path, d))])
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        except FileNotFoundError:
            self.logger.error(f"Directory not found: {root_path}")
            self.classes = []
            self.class_to_idx = {}
        
        # Build file list
        self.video_files = []
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"[cyan]Scanning {len(self.classes)} classes...", total=len(self.classes))
            
            for class_name in self.classes:
                class_dir = os.path.join(root_path, class_name)
                if os.path.isdir(class_dir):
                    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.mp4', '.avi', '.mov'))]
                    self.video_files.extend([(f, self.class_to_idx[class_name]) for f in files])
                progress.update(task, advance=1)
        
        # Random split if not explicitly defined
        if self.split in ['train', 'test']:
            # Shuffle with fixed seed for reproducibility
            random.seed(42)
            random.shuffle(self.video_files)
            
            split_idx = int(len(self.video_files) * 0.8)
            if self.split == 'train':
                self.video_files = self.video_files[:split_idx]
            else:  # 'test'
                self.video_files = self.video_files[split_idx:]
                
        # Transforms - handle numpy arrays properly
        self.transforms = Compose([
            ToTensor(),  # Converts numpy array [H,W,C] to tensor [C,H,W] and scales to [0,1]
            Resize((frame_size, frame_size)),  # Now works on tensors
            CenterCrop(frame_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.logger.info(f"Loaded {len(self.video_files)} video clips from {len(self.classes)} classes")
    
    def __getitem__(self, index: int):
        video_path, label = self.video_files[index] # Define label early for except block
        # Process videos efficiently
        try:
            
            # Check cache
            if self.cache_mode != 'none' and video_path in self.cache:
                return self.cache[video_path], torch.tensor(label)
            
            # Read and process video
            reader = VideoReader(video_path, self.clip_length, self.frame_rate)
            frames = reader.read_video()
            
            if not frames:
                # Fallback to zeros if video reading fails
                self.logger.warning(f"Failed to read video {video_path}, using zero tensor")
                return torch.zeros(self.clip_length, 3, self.frame_size, self.frame_size), torch.tensor(label)
            
            # Apply transforms to each frame
            transformed_frames = []
            for frame in frames:
                # Ensure frame is numpy array and apply transforms
                if isinstance(frame, np.ndarray):
                    # PyAV should provide RGB. OpenCV fallback in VideoReader also converts to RGB.
                    # If frame.shape[-1] == 3: # Color image
                    #     pass # Assume RGB
                    transformed = self.transforms(frame)
                else:
                    # Handle unexpected frame type
                    self.logger.warning(f"Unexpected frame type: {type(frame)} for {video_path}. Attempting conversion.")
                    if isinstance(frame, Image.Image):
                        frame = np.array(frame.convert("RGB")) # Ensure RGB
                    elif hasattr(frame, 'to_ndarray'): # For av.VideoFrame
                        frame = frame.to_ndarray(format='rgb24')
                    elif hasattr(frame, 'numpy'): # For torch.Tensor or similar
                        frame = frame.numpy()
                    else:
                        self.logger.error(f"Cannot convert frame of type {type(frame)} to numpy array for {video_path}. Skipping frame.")
                        continue
                    transformed = self.transforms(frame)
                transformed_frames.append(transformed)
            
            if not transformed_frames:
                self.logger.warning(f"No valid frames extracted from {video_path}, using zero tensor.")
                return torch.zeros(self.clip_length, 3, self.frame_size, self.frame_size), torch.tensor(label)

            # Pad or truncate frames to ensure fixed clip_length
            transformed_frames = self._pad_or_truncate_frames(transformed_frames)

            # Stack frames to create video tensor [T, C, H, W]
            video_tensor = torch.stack(transformed_frames)
            
            # Cache if enabled
            if self.cache_mode == 'memory':
                self.cache[video_path] = video_tensor
                
            return video_tensor, torch.tensor(label)
            
        except Exception as e:
            self.logger.error(f"Error loading video at index {index}: {e}")
            import traceback
            self.logger.debug(f"Full traceback for {video_path}: {traceback.format_exc()}")
            # Return a small zero tensor as fallback
            return torch.zeros(self.clip_length, 3, self.frame_size, self.frame_size), torch.tensor(label)
    
    def __len__(self):
        return len(self.video_files)

    def _pad_or_truncate_frames(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        num_frames = len(frames)
        if num_frames < self.clip_length:
            # Pad with the last frame
            padding = [frames[-1]] * (self.clip_length - num_frames)
            frames.extend(padding)
        elif num_frames > self.clip_length:
            frames = frames[:self.clip_length]
        return frames

#!/usr/bin/env python3
"""
V-JEPA Evaluation Script
"""

import os
import sys
import torch
import argparse

from models.vjepa_tiny import TinyVJEPA
from data.dataset import HMDB51Wrapper
from evals.video_classification import linear_probe_evaluation
from evals.image_classification import cifar10_evaluation
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

console = Console()

def load_model(checkpoint_path, device='cpu'):
    """Load trained V-JEPA model from checkpoint"""
    console.print(f"[cyan]Loading model from {checkpoint_path}[/cyan]")
    
    model = TinyVJEPA(freeze_encoder=False)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'state_dict' in checkpoint:
            state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace('model.', '') if key.startswith('model.') else key
                state_dict[new_key] = value
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        console.print("[green]Model loaded successfully[/green]")
    else:
        console.print("[yellow]No checkpoint provided, using pretrained encoder only[/yellow]")
    
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='V-JEPA Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/hmdb51/subset',
                        help='Path to video dataset')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    console.print(f"[green]Using device: {device}[/green]")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Video evaluation
    if os.path.exists(args.data_dir):
        console.print("\n[bold blue]Video Classification Evaluation[/bold blue]")
        
        train_dataset = HMDB51Wrapper(
            args.data_dir, split='train', clip_length=8, frame_rate=5, frame_size=112
        )
        test_dataset = HMDB51Wrapper(
            args.data_dir, split='test', clip_length=8, frame_rate=5, frame_size=112
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
        
        num_classes = len(train_dataset.classes)
        
        video_acc = linear_probe_evaluation(
            model, train_loader, test_loader, num_classes, device
        )
        
        console.print(f"[bold green]Video Classification Accuracy: {video_acc:.2f}%[/bold green]")
    
    # Image evaluation
    console.print("\n[bold blue]Image Classification Evaluation[/bold blue]")
    image_acc = cifar10_evaluation(model, device, args.batch_size)
    console.print(f"[bold green]CIFAR-10 Accuracy: {image_acc:.2f}%[/bold green]")

if __name__ == '__main__':
    main()

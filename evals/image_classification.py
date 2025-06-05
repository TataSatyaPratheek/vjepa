import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score
import logging
from rich.progress import track
from rich.console import Console

console = Console()
logger = logging.getLogger("eval_image")

class ImageClassificationEvaluator:
    """Image classification evaluation using V-JEPA features"""
    def __init__(self, model, num_classes, freeze_encoder=True, device='cpu'):
        self.model = model
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.device = device
        
        embed_dim = model.encoder.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(embed_dim, num_classes)
        ).to(device)
        
        if freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False
        
        console.print(f"[green]Initialized image classifier with {num_classes} classes[/green]")
        
    def train_classifier(self, train_loader, val_loader=None, epochs=10, lr=1e-3):
        """Train classification head on image features"""
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(), 
            lr=lr, 
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            self.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(track(
                train_loader, description=f"Epoch {epoch+1}/{epochs}"
            )):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Convert single images to video format [B, 1, C, H, W]
                if images.ndim == 4:
                    images = images.unsqueeze(1)
                
                with torch.no_grad():
                    features = self._extract_features(images)
                
                logits = self.classifier(features)
                loss = F.cross_entropy(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
            scheduler.step()
            
            train_acc = 100. * train_correct / train_total
            avg_loss = train_loss / len(train_loader)
            
            console.print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={train_acc:.2f}%")
            
            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                if val_acc > best_acc:
                    best_acc = val_acc
        
        return best_acc
    
    def evaluate(self, test_loader):
        """Evaluate on test images"""
        self.classifier.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in track(test_loader, description="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if images.ndim == 4:
                    images = images.unsqueeze(1)
                
                features = self._extract_features(images)
                logits = self.classifier(features)
                _, predicted = logits.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        console.print(f"[bold green]Test Accuracy: {accuracy*100:.2f}%[/bold green]")
        
        return accuracy * 100
    
    def _extract_features(self, images):
        """Extract features from images using video encoder"""
        features = self.model(images, return_all_features=False)  # [B, 1, D]
        features = features.squeeze(1)  # [B, D]
        return features

def cifar10_evaluation(model, device='cpu', batch_size=64):
    """CIFAR-10 evaluation following V-JEPA protocol"""
    console.print("[bold blue]Starting CIFAR-10 Evaluation[/bold blue]")
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    evaluator = ImageClassificationEvaluator(
        model, num_classes=10, freeze_encoder=True, device=device
    )
    
    best_acc = evaluator.train_classifier(train_loader, test_loader, epochs=10)
    final_acc = evaluator.evaluate(test_loader)
    
    return final_acc

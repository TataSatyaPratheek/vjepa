import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import logging
from rich.progress import track
from rich.console import Console

console = Console()
logger = logging.getLogger("eval_video")

class VideoClassificationEvaluator:
    """
    Frozen evaluation for video classification following V-JEPA patterns
    """
    def __init__(self, model, num_classes, freeze_encoder=True, device='cpu'):
        self.model = model
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.device = device
        
        # Create classification head
        embed_dim = model.encoder.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(embed_dim, num_classes)
        ).to(device)
        
        if freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False
        
        console.print(f"[green]Initialized video classifier with {num_classes} classes[/green]")
        
    def train_classifier(self, train_loader, val_loader=None, epochs=10, lr=1e-3):
        """Train classification head on frozen features"""
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
            
            for batch_idx, (videos, labels) in enumerate(track(
                train_loader, description=f"Epoch {epoch+1}/{epochs}"
            )):
                videos, labels = videos.to(self.device), labels.to(self.device)
                
                # Extract features with frozen encoder
                with torch.no_grad():
                    features = self._extract_features(videos)
                
                # Forward through classifier
                logits = self.classifier(features)
                loss = F.cross_entropy(logits, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
            scheduler.step()
            
            train_acc = 100. * train_correct / train_total
            avg_loss = train_loss / len(train_loader)
            
            console.print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={train_acc:.2f}%")
            
            # Validation
            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                if val_acc > best_acc:
                    best_acc = val_acc
        
        return best_acc
    
    def evaluate(self, test_loader):
        """Evaluate classifier on test set"""
        self.classifier.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in track(test_loader, description="Evaluating"):
                videos, labels = videos.to(self.device), labels.to(self.device)
                
                features = self._extract_features(videos)
                logits = self.classifier(features)
                _, predicted = logits.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        console.print(f"[bold green]Test Accuracy: {accuracy*100:.2f}%[/bold green]")
        
        return accuracy * 100
    
    def _extract_features(self, videos):
        """Extract features from videos using frozen encoder"""
        B, T = videos.shape[:2]
        features = self.model(videos, return_all_features=False)  # [B, T, D]
        features = features.mean(dim=1)  # [B, D] - Global average pooling over time
        return features

def linear_probe_evaluation(model, train_loader, test_loader, num_classes, device='cpu'):
    """Perform linear probe evaluation following V-JEPA protocol"""
    console.print("[bold blue]Starting Linear Probe Evaluation[/bold blue]")
    
    evaluator = VideoClassificationEvaluator(
        model, num_classes, freeze_encoder=True, device=device
    )
    
    best_acc = evaluator.train_classifier(train_loader, test_loader, epochs=10, lr=1e-3)
    final_acc = evaluator.evaluate(test_loader)
    
    return final_acc

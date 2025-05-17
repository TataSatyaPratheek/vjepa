import os
import torch
from torch.optim import AdamW
from torch.nn import functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from utils.masking import create_tube_mask
from data.dataset import HMDB51Wrapper
from models.vjepa_tiny import TinyVJEPA
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.multiprocessing as mp

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

class VJEPATrainer(L.LightningModule):
    def __init__(self, model, config=None):
        super().__init__()
        self.model = model
        self.automatic_optimization = False
        self.config = config or {}
        self.save_hyperparameters(ignore=['model'])
        
        # Training configuration
        self.lr = self.config.get('lr', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 1e-2)
        self.batch_size = self.config.get('batch_size', 2)
        self.mask_ratio = self.config.get('mask_ratio', 0.75)
        
    def on_fit_start(self):
        # Make sure MPS backend is used if available
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS acceleration!")
        
    def training_step(self, batch):
        opt = self.optimizers()
        video, _ = batch  # [B, T, C, H, W]
        
        # Generate tube mask
        mask = create_tube_mask(video.shape, self.mask_ratio)
        
        # Mask video using memory-efficient operations
        masked_video = video.clone()
        masked_video[mask == 0] = 0

        # Free memory
        mask = None
        
        # MPS-specific mixed precision - use float16 instead of bfloat16 for M1
        with torch.autocast(device_type='mps', dtype=torch.float16):
            pred_features, _ = self.model(masked_video, return_all_features=True)
            
            # Free memory
            masked_video = None
            
            with torch.no_grad():
                # Process in smaller chunks to handle memory constraints
                batch_size, seq_len = video.shape[:2]
                chunk_size = min(4, batch_size * seq_len)
                target_features_list = []
                
                flat_video = video.flatten(0, 1)
                for i in range(0, flat_video.shape[0], chunk_size):
                    chunk = flat_video[i:i+chunk_size]
                    chunk_features = self.model.encoder(chunk).last_hidden_state
                    target_features_list.append(chunk_features)
                    
                target_features = torch.cat(target_features_list, dim=0)
                target_features = target_features.unflatten(0, video.shape[:2])
                
                # Free memory
                flat_video = None
                
            # Use cosine similarity loss which is more effective and memory-efficient
            loss = 1 - F.cosine_similarity(
                pred_features.reshape(-1, pred_features.shape[-1]),
                target_features.reshape(-1, target_features.shape[-1]),
                dim=1
            ).mean()
        
        # Manual optimization with gradient clipping
        opt.zero_grad()
        self.manual_backward(loss)
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()
        
        # More memory-efficient logging
        self.log("train_loss", loss.item(), prog_bar=True)
        
        # Force clean memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        return loss

    def configure_optimizers(self):
        return AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

def train():
    # Use much smaller batch size for 8GB M1
    config = {
        'lr': 5e-5,  # Lower learning rate for stability
        'weight_decay': 1e-2,
        'batch_size': 1,  # Use batch size of 1 for 8GB MacBook Air
        'mask_ratio': 0.75,
        'num_workers': 1  # Reduce workers for MacBook
    }
    
    model = TinyVJEPA(freeze_encoder=True)
    
    # Set up checkpointing for resuming training
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="vjepa-{epoch:02d}-{train_loss:.3f}",
        monitor="train_loss",
        save_last=True,
        save_top_k=2,
        mode="min",
        every_n_train_steps=100
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir="logs", name="vjepa")
    
    # Configure trainer with M1-specific settings
    trainer = L.Trainer(
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        max_epochs=30,  # Reduce epochs
        precision="16-mixed",  # Use float16 for M1
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        accumulate_grad_batches=4,  # Accumulate gradients for effective batch size of 4
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Define data directory from environment or use default
    data_dir = os.environ.get("HMDB51_DIR", "data/hmdb51/subset")
    
    # Prepare datasets
    train_dataset = HMDB51Wrapper(data_dir, split='train', clip_length=8, frame_rate=5)
    val_dataset = HMDB51Wrapper(data_dir, split='test', clip_length=8, frame_rate=5)
    
    # Create model wrapper
    model_wrapper = VJEPATrainer(model, config)
    
    # Use smaller prefetch factor and pin memory for M1
    train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            num_workers=config['num_workers'], 
            pin_memory=False,  # Pin memory can cause issues on M1
            persistent_workers=False  # Disable for stability
        )
    
    val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            num_workers=config['num_workers']
    )
    
    # Start training
    trainer.fit(model_wrapper, train_loader, val_loader)

if __name__ == "__main__":
    train()

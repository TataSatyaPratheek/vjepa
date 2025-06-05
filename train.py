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
import torch.multiprocessing as mp
import logging
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn
from rich.console import Console
from rich import print as rprint
from rich.panel import Panel
from pathlib import Path
import time
import sys
from colorama import Fore, Style
import gc
import numpy as np
import math

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

log = logging.getLogger("vjepa")
console = Console()

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0):
    """Cosine scheduler for momentum parameter following official V-JEPA"""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

try:
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

class TrainingProgressCallback(L.Callback):
    """Custom callback for detailed progress tracking"""
    def __init__(self):
        super().__init__()
        self.train_start_time = None
        self.epoch_start_time = None
        self.step_start_time = None
        self.last_memory_log = 0
        self.memory_log_interval = 10  # Log memory every 10 steps
        
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        rprint(Panel.fit(
            f"[bold green]Starting training with {trainer.max_epochs} epochs[/bold green]\n"
            f"[cyan]Model: {pl_module.__class__.__name__}[/cyan]\n"
            f"Using {Fore.YELLOW}MPS acceleration{Style.RESET_ALL}" if torch.backends.mps.is_available() 
            else f"Using {Fore.RED}CPU{Style.RESET_ALL}"
        ))
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        epoch = trainer.current_epoch + 1
        rprint(f"\n{Fore.GREEN}[Epoch {epoch}/{trainer.max_epochs}]{Style.RESET_ALL} Starting...")
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step_start_time = time.time()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step_time = time.time() - self.step_start_time
        step = trainer.global_step
        epoch = trainer.current_epoch + 1
        loss = trainer.callback_metrics.get("train_loss", torch.tensor(0.0)).item()
        
        # Only log memory usage periodically to save overhead
        memory_info = ""
        if step - self.last_memory_log >= self.memory_log_interval:
            if torch.backends.mps.is_available():
                # For MPS backend
                used_memory = torch.mps.current_allocated_memory() / (1024 ** 2)  # MB
                memory_info = f"{Fore.MAGENTA}Memory: {used_memory:.1f}MB{Style.RESET_ALL} | "
                self.last_memory_log = step
                # Force garbage collection
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        
        # Calculate ETA
        elapsed_time = time.time() - self.train_start_time
        progress_fraction = (epoch - 1 + (batch_idx / len(trainer.train_dataloader))) / trainer.max_epochs
        if progress_fraction > 0:
            eta_seconds = elapsed_time * (1/progress_fraction - 1)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            eta_info = f"{Fore.CYAN}ETA: {eta_str}{Style.RESET_ALL} | "
        else:
            eta_info = ""
            
        # Print colorful progress
        print(f"\r{Fore.GREEN}[Epoch {epoch}/{trainer.max_epochs}]{Style.RESET_ALL} "
              f"Step: {step} | {memory_info}{eta_info}"
              f"{Fore.BLUE}Loss: {loss:.4f}{Style.RESET_ALL} | "
              f"{Fore.YELLOW}Step time: {step_time:.2f}s{Style.RESET_ALL}", end="")
        sys.stdout.flush()
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        epoch = trainer.current_epoch + 1
        loss = trainer.callback_metrics.get("train_loss", torch.tensor(0.0)).item()
        
        rprint(f"\n{Fore.GREEN}[Epoch {epoch}/{trainer.max_epochs} Complete]{Style.RESET_ALL} "
               f"Loss: {loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Force memory cleanup at the end of epoch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    def on_train_end(self, trainer, pl_module):
        total_time = time.time() - self.train_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        rprint(Panel.fit(
            f"[bold green]Training Complete![/bold green]\n"
            f"Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n"
            f"Final loss: {trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item():.4f}"
        ))

class VJEPATrainer(L.LightningModule):
    def __init__(self, model, config=None):
        super().__init__()
        self.model = model
        self.automatic_optimization = False
        self.config = config or {}
        # Save hyperparameters, but ignore large objects
        self.save_hyperparameters(ignore=['model'])
        
        # Setup logger
        self.log_dict = {}
        
        # Training configuration
        self.lr = self.config.get('lr', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 1e-2)
        self.batch_size = self.config.get('batch_size', 2)
        self.mask_ratio = self.config.get('mask_ratio', 0.75)
        self.accum_grad_batches = self.config.get('accum_grad_batches', 4)
        self.grad_clip_val = self.config.get('grad_clip_val', 1.0)

        # V-JEPA specific parameters
        self.momentum_start = self.config.get('momentum_start', 0.996)
        self.momentum_end = self.config.get('momentum_end', 1.0)

        # Initialize momentum scheduler
        max_epochs = self.config.get('max_epochs', 30)
        # Placeholder, will be updated in on_train_start based on actual dataloader length
        steps_per_epoch_initial = self.config.get('steps_per_epoch_initial_guess', 100)
        self.momentum_schedule = cosine_scheduler(
            self.momentum_start, self.momentum_end,
            max_epochs, steps_per_epoch_initial
        )
        
        log.info(f"Initialized VJEPATrainer with lr={self.lr}, batch_size={self.batch_size}")
        log.info(f"Momentum schedule (initial): {self.momentum_start} -> {self.momentum_end}")

    def on_train_start(self):
        """Update momentum scheduler with actual steps per epoch and log device"""
        # Make sure MPS backend is used if available
        if torch.backends.mps.is_available():
            log.info(f"{Fore.GREEN}Using MPS acceleration!{Style.RESET_ALL}")
        else:
            log.warning(f"{Fore.YELLOW}MPS not available, using CPU{Style.RESET_ALL}")
        steps_per_epoch = len(self.trainer.train_dataloader) // self.accum_grad_batches
        max_epochs = self.trainer.max_epochs
        self.momentum_schedule = cosine_scheduler(
            self.momentum_start, self.momentum_end,
            max_epochs, steps_per_epoch
        )
        log.info(f"Updated momentum scheduler for {steps_per_epoch} effective steps/epoch, {max_epochs} epochs")
        
    def training_step(self, batch, batch_idx):
        """Training step with memory optimization"""
        opt = self.optimizers()
        video, _ = batch  # [B, T, C, H, W]
        
        # Print shape info on first batch
        if self.global_step == 0:
            log.info(f"Video shape: {video.shape}")
        
        # Generate context and target masks following V-JEPA pattern
        try:
            masks_context_bool = create_tube_mask(video.shape, self.mask_ratio) # boolean mask
            # Target mask can be different, e.g., smaller or sparser, or even full for some VJEPA variants
            masks_target_bool = create_tube_mask(video.shape, 0.1) # Example: smaller mask ratio for target
        except Exception as e:
            log.error(f"Error creating mask: {e}")
            masks_context_bool = torch.ones_like(video[0,0]).bool() # Fallback, assuming C,H,W for mask
            masks_target_bool = torch.ones_like(video[0,0]).bool()
        
        # Apply masks
        # For V-JEPA, context is masked, target is usually the original unmasked video for encoding
        video_context = video.clone()
        # create_tube_mask returns a mask for [T, C, H, W] or [T, H, W]
        # Assuming create_tube_mask returns [B, T, 1, H, W] or similar, adjust if needed
        # If masks_context_bool is [B, T, H, W], unsqueeze for channel: masks_context_bool.unsqueeze(2)
        video_context[~masks_context_bool.unsqueeze(2).expand_as(video)] = 0 # Apply mask where it's False
        video_target = video # Target video is typically unmasked for the target encoder

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.backends.mps.is_available() and self.global_step % 10 == 0:
            torch.mps.empty_cache()
        
        # Try with autocast, fall back if it fails
        try:
            # MPS-specific mixed precision - use float16 instead of bfloat16 for M1
            device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            dtype = torch.float16

            with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type != 'cpu'):
                # Forward pass with context and target
                # The model's forward should handle x_target and masks_target
                pred_features, context_features, target_features = self.model(
                    x=video_context,  # Masked video for context encoder
                    x_target=video_target, # Original video for target encoder
                    # masks_context is implicitly handled by video_context being pre-masked
                    # masks_target is for the loss calculation, not for target encoder input
                    return_all_features=True
                )
                video_context = None # Free memory

                # V-JEPA loss: predict target features from context
                if target_features is not None:
                    # Ensure masks_target_bool is correctly shaped for feature masking
                    # pred_features/target_features are [B, T, HiddenDim]
                    # masks_target_bool is [B, T, H, W] from create_tube_mask
                    # We need a mask of shape [B, T] for the features.
                    # This usually means selecting which *patches* or *frames* in the target to predict.
                    # For simplicity, let's assume masks_target_bool can be reduced to [B,T]
                    # For V-JEPA, the loss is often computed over *patches* of the target.
                    # The model's forward pass should ideally return features aligned with these masks.
                    # If pred_features are [B, T, NumPatches, Dim] and target_features are similar,
                    # then masks_target needs to be [B, T, NumPatches]
                    # For now, assuming features are [B,T,Dim] and mask is [B,T]
                    # This part needs careful alignment with how TinyVJEPA handles masks and returns features.
                    # Assuming masks_target_bool from create_tube_mask is [B, T, H, W]
                    # We need to convert this to a mask for features [B, T, Dim]
                    # A common V-JEPA approach is to mask *target patches* for the loss.
                    # The current TinyVJEPA forward returns CLS tokens [B, T, HiddenSize]
                    # So, a mask of [B, T] is appropriate here.
                    # Let's assume masks_target_bool is effectively a frame-level mask [B,T] for simplicity here.
                    # If create_tube_mask returns [B,T,H,W], we can take .any() over H,W.
                    target_frames_mask = masks_target_bool.any(dim=(-1,-2)) # [B,T]

                    loss = F.mse_loss(
                        pred_features[target_frames_mask],
                        target_features[target_frames_mask]
                    )
                else: # Fallback if target_features are not available (e.g. model error)
                    log.warning("Target features are None, using simplified self-supervised loss.")
                    loss = F.mse_loss(pred_features, context_features) # Or another appropriate loss
        except Exception as e:
            log.error(f"Error in autocast block: {e}. Trying without autocast...")
            # Fallback without autocast
            pred_features, context_features, target_features = self.model(
                x=video_context, x_target=video_target, return_all_features=True
            )
            video_context = None
            if target_features is not None:
                target_frames_mask = masks_target_bool.any(dim=(-1,-2))
                loss = F.mse_loss(pred_features[target_frames_mask], target_features[target_frames_mask])
            else:
                loss = F.mse_loss(pred_features, context_features) if context_features is not None else torch.tensor(0.0, device=pred_features.device)

        # Update target encoder with momentum
        current_step_in_epoch = self.global_step % (len(self.trainer.train_dataloader) // self.accum_grad_batches)
        current_epoch_effective_steps = len(self.trainer.train_dataloader) // self.accum_grad_batches
        schedule_idx = self.trainer.current_epoch * current_epoch_effective_steps + current_step_in_epoch

        if schedule_idx < len(self.momentum_schedule):
            momentum = self.momentum_schedule[schedule_idx]
            self.model.update_target_encoder(momentum)
        else: # Should not happen if schedule is correctly sized
            momentum = self.momentum_end 
            self.model.update_target_encoder(momentum)
            if self.global_step == 0 : # Log only once if this occurs
                 log.warning(f"Momentum schedule index {schedule_idx} out of bounds (len {len(self.momentum_schedule)}). Using momentum_end.")

        # Handle gradient accumulation
        is_accumulating = (batch_idx + 1) % self.accum_grad_batches != 0
        
        try:
            # Manual optimization with gradient clipping
            opt.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            
            if not is_accumulating or (batch_idx + 1) == self.trainer.num_training_batches:
                # Clip gradients to prevent instability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
                opt.step()
                
                # Log step completion with accumulated batches
                if batch_idx % 10 == 0 or (batch_idx + 1) == self.trainer.num_training_batches:
                    log.debug(f"Step completed: {batch_idx+1}/{self.trainer.num_training_batches} "
                             f"(Accumulated {self.accum_grad_batches} batches)")
        except Exception as e:
            log.error(f"Error in optimization step: {e}")
            # Try to continue with next batch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        # Memory-efficient logging
        current_momentum = self.momentum_schedule[schedule_idx] if schedule_idx < len(self.momentum_schedule) else self.momentum_end
        metrics = {
            "train_loss": loss.item(),
            "momentum": current_momentum
        }
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=False, logger=True) # Log on step for Tensorboard
        
        # Force clean memory
        if torch.backends.mps.is_available() and batch_idx % 5 == 0:
            torch.mps.empty_cache()
            
        return loss

    def configure_optimizers(self):
        """Configure optimizers with warmup and weight decay"""
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        # Group parameters by weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        params_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(
            params_groups,
            lr=self.lr, 
            betas=(0.9, 0.999), 
            eps=1e-8
        )

def train():
    """Main training function with error handling and recovery"""
    try:
        console.print(Panel.fit(
            "[bold cyan]Starting Video JEPA Training[/bold cyan]\n"
            "[green]Optimized for M1 MacBook Air 8GB[/green]"
        ))

        # Create output directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
        # Use much smaller batch size for 8GB M1
        config = {
            'lr': 5e-5,  # Lower learning rate for stability
            'weight_decay': 1e-2,
            'batch_size': 1,  # Use batch size of 1 for 8GB MacBook Air
            'mask_ratio': 0.75,
            'num_workers': 1,  # Reduce workers for MacBook
            'accum_grad_batches': 4,  # Accumulate gradients for effective batch size of 4
            'grad_clip_val': 1.0,
            'max_epochs': 30,
            # 'steps_per_epoch_initial_guess' will be implicitly handled by on_train_start
        }
        
        console.print("[cyan]Initializing model...[/cyan]")
        model = TinyVJEPA(freeze_encoder=True)
        
        # Set up checkpointing for resuming training
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="vjepa-{epoch:02d}-{train_loss:.3f}",
            monitor="train_loss",
            save_last=True,
            save_top_k=2,
            mode="min",
            every_n_train_steps=100,
            save_on_train_epoch_end=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        progress_callback = TrainingProgressCallback()
        logger = TensorBoardLogger(save_dir="logs", name="vjepa")
        
        console.print("[cyan]Setting up trainer...[/cyan]")
        # Configure trainer with M1-specific settings
        trainer = L.Trainer(
            accelerator="mps" if torch.backends.mps.is_available() else "cpu",
            devices=1,
            max_epochs=config['max_epochs'],
            precision="16-mixed",  # Use float16 for M1
            callbacks=[checkpoint_callback, lr_monitor, progress_callback],
            logger=logger,
            log_every_n_steps=10,
            enable_progress_bar=False,  # Custom progress tracking
            enable_model_summary=True,
            deterministic=False,  # For better performance
            benchmark=True
        )
        
        # Define data directory from environment or use default
        data_dir = os.environ.get("HMDB51_DIR", "data/hmdb51/subset")
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            console.print(f"[bold red]Error: Data directory {data_dir} not found![/bold red]")
            console.print("[yellow]Please set the HMDB51_DIR environment variable to your dataset path.[/yellow]")
            return
        
        console.print("[cyan]Loading datasets...[/cyan]")
        # Prepare datasets with memory cache for M1
        train_dataset = HMDB51Wrapper(
            data_dir, 
            split='train', 
            clip_length=8, 
            frame_rate=5,
            frame_size=112,
            cache_mode='none'  # Set to 'memory' for faster training if you have enough RAM
        )
        
        val_dataset = HMDB51Wrapper(
            data_dir, 
            split='test', 
            clip_length=8, 
            frame_rate=5,
            frame_size=112
        )
        
        console.print("[cyan]Creating model wrapper...[/cyan]")
        # Create model wrapper
        model_wrapper = VJEPATrainer(model, config)
        
        console.print("[cyan]Setting up data loaders...[/cyan]")
        # Use smaller prefetch factor and pin memory for M1
        train_loader = DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                num_workers=config['num_workers'], 
                pin_memory=False,  # Pin memory can cause issues on M1
                persistent_workers=False,  # Disable for stability
                shuffle=True
            )
        
        val_loader = DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                num_workers=config['num_workers'],
                shuffle=False
        )
    
        console.print("[bold green]Starting training...[/bold green]")
        # Start training
        trainer.fit(model_wrapper, train_loader, val_loader)
        
        console.print("[bold green]Training complete! Model saved to checkpoints/vjepa-last.ckpt[/bold green]")
        
    except KeyboardInterrupt:
        console.print("[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error during training: {e}[/bold red]")
        import traceback
        console.print_exception()

if __name__ == "__main__":
    train()

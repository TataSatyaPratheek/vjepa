import pytorch_lightning as pl
from torch.optim import AdamW
from torch.nn import functional as F

class VJEPATrainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.automatic_optimization = False
        
    def training_step(self, batch):
        opt = self.optimizers()
        video, _ = batch  # [B, T, C, H, W]
        
        # Generate tube mask
        mask = create_tube_mask(video.shape)
        
        # Mask video using in-place operations
        masked_video = video.clone()
        masked_video[mask == 0] = 0
        
        # Mixed precision forward
        with torch.autocast(device_type='mps', dtype=torch.bfloat16):
            pred_features = self.model(masked_video)
            with torch.no_grad():
                target_features = self.model.encoder(video.flatten(0, 1)).last_hidden_state.unflatten(0, video.shape[:2])
            loss = F.mse_loss(pred_features, target_features)
        
        # Manual optimization with gradient clipping
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.model.predictor.parameters(), lr=1e-4)

def train():
    model = TinyVJEPA()
    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=50,
        precision="bf16-mixed",
        gradient_clip_val=0.5,
        enable_checkpointing=True,
        enable_model_summary=True
    )
    
    train_dataset = HMDB51Wrapper("data/hmdb51/subset", split='train')
    val_dataset = HMDB51Wrapper("data/hmdb51/subset", split='test')
    
    trainer.fit(
        model, 
        train_dataloaders=torch.utils.data.DataLoader(
            train_dataset, batch_size=2, num_workers=2, persistent_workers=True
        ),
        val_dataloaders=torch.utils.data.DataLoader(
            val_dataset, batch_size=2, num_workers=2
        )
    )

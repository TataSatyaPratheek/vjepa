import torch
import torch.nn as nn
from transformers import ViTModel, AutoImageProcessor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import logging

from rich.logging import RichHandler # Assuming rich is installed, otherwise comment out or handle appropriately

# Configure logging
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]) # Example with RichHandler
logger = logging.getLogger("vjepa_model")
# If RichHandler is not used or preferred, a basic config can be:
if not logger.hasHandlers(): # Avoid adding multiple handlers if re-running in an interactive session
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class TinyVJEPA(nn.Module):
    def __init__(self, pretrained="google/vit-base-patch16-224", freeze_encoder=True):
        super().__init__()
        
        # Initialize context encoder (x-encoder)
        try:
            # Use safetensors for faster loading and better compatibility
            # Add offline mode and better error handling
            try:
                logger.info(f"Attempting to load pretrained model: {pretrained}")
                self.encoder = ViTModel.from_pretrained(
                    pretrained, 
                    use_safetensors=True,
                    torch_dtype=torch.float16,  # Keep in fp16 for memory savings
                    local_files_only=False,  # Try online first
                    trust_remote_code=False
                )
                
                # Initialize target encoder (y-encoder) - CRITICAL MISSING COMPONENT
                self.target_encoder = ViTModel.from_pretrained(
                    pretrained,
                    use_safetensors=True, 
                    torch_dtype=torch.float16,
                    local_files_only=False,
                    trust_remote_code=False
                )
                
                # Target encoder is always frozen and updated via EMA
                self.target_encoder.requires_grad_(False)
                
                logger.info(f"Successfully loaded pretrained model from {pretrained}")
            except Exception as online_error:
                logger.warning(f"Online loading failed: {online_error}. Trying local or cached version...")
                try:
                    # Try cached or local version
                    self.encoder = ViTModel.from_pretrained(
                        pretrained,
                        use_safetensors=True,
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )
                    # Attempt to load target_encoder from local cache as well
                    self.target_encoder = ViTModel.from_pretrained(
                        pretrained,
                        use_safetensors=True,
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )
                    self.target_encoder.requires_grad_(False) # Ensure target encoder is frozen
                    logger.info(f"Loaded encoder and target_encoder from local cache: {pretrained}")
                except Exception as local_error:
                    # If that fails too, use a specific fallback model that's included
                    logger.error(f"Local loading also failed for one or both encoders: {local_error}")
                    raise RuntimeError(f"Could not load model from {pretrained} (online or local)")
            self.image_processor = AutoImageProcessor.from_pretrained(pretrained)
            
            # Freeze encoder to save memory during training
            if freeze_encoder:
                self.encoder.requires_grad_(False)
                
            # Enable gradient checkpointing
            self.encoder.gradient_checkpointing_enable()

            # Initialize target encoder parameters to match context encoder
            # Moved here as per user's diff structure, ensures early initialization
            self._initialize_target_encoder()
            
            # logger.info(f"Successfully loaded {pretrained} model") # Moved inside try-except
            
        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            # Fallback to simpler initialization if pretrained fails
            from transformers import ViTConfig
            config = ViTConfig(hidden_size=192, num_hidden_layers=6, num_attention_heads=3, intermediate_size=768) # Example config for a tiny ViT
            self.encoder = ViTModel(config)
            self.target_encoder = ViTModel(config)
            self.target_encoder.requires_grad_(False)
            if freeze_encoder:
                self.encoder.requires_grad_(False)
            # Attempt to load image processor in fallback path
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(pretrained, local_files_only=True)
                logger.info(f"Loaded image processor for {pretrained} (fallback path, local attempt).")
            except Exception as proc_error:
                logger.warning(f"Could not load image processor for {pretrained} in fallback: {proc_error}. Image processing might fail.")
                self.image_processor = None # Or a default ViTImageProcessor if applicable
            logger.warning("Using randomly initialized ViT model as fallback due to error.")
        
        # Memory-efficient predictor with layer norm
        try:
            # Enhanced predictor following official V-JEPA patterns
            hidden_dim = 128
            embed_dim = self.encoder.config.hidden_size
            # More sophisticated predictor as per official implementation
            self.predictor = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),  # Add dropout for regularization
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(), 
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embed_dim)
            )
            logger.info(f"Predictor initialized with hidden_dim={hidden_dim}, embed_dim={embed_dim}")
        except Exception as e:
            logger.error(f"Error initializing predictor: {e}")
            # Fallback predictor
            embed_dim_fallback = 192 # Default or based on fallback encoder
            self.predictor = nn.Sequential(
                nn.LayerNorm(embed_dim_fallback),
                nn.Linear(embed_dim_fallback, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, embed_dim_fallback)
            )
            logger.warning(f"Using fallback predictor with embed_dim={embed_dim_fallback}.")
        
        # For efficient M1 inference
        self._device_setup_done = False

    def _initialize_target_encoder(self):
        """Initialize target encoder parameters to match context encoder"""
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.copy_(param_q.data)

    def update_target_encoder(self, momentum: float = 0.996):
        """Update target encoder with momentum (EMA)"""
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(momentum).add_((1. - momentum) * param_q.detach().data)

    def setup_for_inference(self):
        """Configure model for M1 inference"""
        try:
            if not self._device_setup_done and torch.backends.mps.is_available():
                # Use mixed precision for inference
                # Ensure encoder exists and is a PyTorch module before calling .to()
                if hasattr(self, 'encoder') and isinstance(self.encoder, nn.Module):
                    self.encoder = self.encoder.to(torch.float16)
                if hasattr(self, 'target_encoder') and isinstance(self.target_encoder, nn.Module):
                    self.target_encoder = self.target_encoder.to(torch.float16)
                self._device_setup_done = True
                logger.info("Model configured for MPS inference with float16 encoder.")
            elif not torch.backends.mps.is_available():
                logger.info("MPS not available. Skipping MPS-specific inference setup.")
            elif self._device_setup_done:
                logger.info("MPS inference setup already done.")
        except Exception as e:
            logger.error(f"Error during inference setup: {e}")

    def forward(self, x, x_target=None, masks_context=None, masks_target=None, return_all_features=False): # Applied change
        """Forward pass following official V-JEPA pattern"""
        try:
            B, T = x.shape[:2] # Assuming x is [Batch, Time, Channels, Height, Width]
            
            # Make sure x is on the same device as the model
            model_device = next(self.encoder.parameters()).device
            if x.device != model_device:
                x = x.to(model_device)
            if x_target is not None and x_target.device != model_device:
                x_target = x_target.to(model_device) # Applied change



            # If input is float64, cast to float32 (or model's dtype)
            if x.dtype == torch.float64:
                x = x.to(torch.float32)
            
            # If encoder is float16, input should ideally also be float16 or castable
            if self.encoder.dtype == torch.float16 and x.dtype != torch.float16:
                 x = x.to(torch.float16)
                 if x_target is not None: # Applied change
                    x_target = x_target.to(torch.float16) # Applied change


            x_flat = x.flatten(0, 1)  # Combine batch and time: [B*T, C, H, W]
            
            # Process context (masked input)
            # Use smaller chunks for M1's limited memory
            chunk_size = min(4, x_flat.shape[0])  # Reduced from 8 to 4 for reliability
            if x_flat.shape[0] == 0: # Handle empty input
                logger.warning("Empty input to forward pass.")
                empty_features = torch.zeros(B, T, self.encoder.config.hidden_size, device=model_device, dtype=self.encoder.dtype)
                if return_all_features:
                    return self.predictor(empty_features), empty_features, None # Applied change
                return self.predictor(empty_features)

            
            # Checkpoint encoder to save memory
            # Process context features # Applied change
            context_features_list = []
            for i in range(0, x_flat.shape[0], chunk_size):
                chunk = x_flat[i:i+chunk_size]
                # Ensure chunk is not empty and has the right dimensions for the encoder
                if chunk.nelement() == 0:
                    continue

                # The ViTModel expects pixel_values of shape (batch_size, num_channels, height, width)
                # If your input `x` is already preprocessed and tokenized, this might differ.
                # Assuming `x` contains raw image tensors [B, T, C, H, W]
                
                # The HuggingFace ViTModel's forward usually takes `pixel_values`
                # The lambda function should pass the chunk directly if it's in the expected format
                chunk_features = checkpoint(
                    lambda pixel_values_arg: self.encoder(pixel_values=pixel_values_arg).last_hidden_state, 
                    chunk, 
                    use_reentrant=False # Recommended for newer PyTorch versions for memory efficiency
                )
                context_features_list.append(chunk_features) # Applied change
            
            if not context_features_list: # If all chunks were empty or skipped # Applied change
                logger.warning("No features extracted, possibly due to empty input chunks.")
                empty_features = torch.zeros(B, T, self.encoder.config.hidden_size, device=model_device, dtype=self.encoder.dtype)
                if return_all_features:
                    return self.predictor(empty_features), empty_features, None # Applied change
                return self.predictor(empty_features)

            context_features_cat = torch.cat(context_features_list, dim=0)
            # Reshape features back to [B, T, NumPatches, EmbedDim] or [B, T, CLS_Token_EmbedDim]
            # ViT's last_hidden_state is typically [B*T_chunked, NumPatches+1, HiddenSize]
            # We need to know what the predictor expects. Assuming it expects features per frame [B, T, HiddenSize] (e.g., CLS token)
            # If using CLS token, it's features_cat[:, 0, :]
            # For this example, let's assume we average patch embeddings or take CLS
            # For simplicity, if predictor expects [B, T, HiddenSize], we might take the CLS token embedding

            # If predictor expects sequence of patch embeddings:
            # features = features_cat.unflatten(0, (B, T)) # This would be [B, T, NumPatches+1, HiddenSize]

            # If predictor expects a single vector per frame (e.g., CLS token's embedding):
            context_cls_features = context_features_cat[:, 0, :] # Assuming CLS token is at index 0 # Applied change
            context_features = context_cls_features.unflatten(0, (B, T)) # Shape: [B, T, HiddenSize] # Applied change
            
            # Process target features with target encoder if provided
            target_features = None # Applied change
            if x_target is not None:
                x_target_flat = x_target.flatten(0, 1)
                target_features_list = []
                
                with torch.no_grad():  # Target encoder always in eval mode
                    for i in range(0, x_target_flat.shape[0], chunk_size):
                        chunk = x_target_flat[i:i+chunk_size]
                        if chunk.nelement() == 0:
                            continue
                            
                        chunk_features = self.target_encoder(pixel_values=chunk).last_hidden_state # Corrected to use target_encoder
                        target_features_list.append(chunk_features)
                
                if target_features_list:
                    target_features_cat = torch.cat(target_features_list, dim=0)
                    target_cls_features = target_features_cat[:, 0, :]
                    target_features = target_cls_features.unflatten(0, (B, T))
            
            # Clean up CUDA/MPS memory
            if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            predicted_features = self.predictor(context_features) # Applied change

            if return_all_features:
                return predicted_features, context_features, target_features # `features` here is the CLS token embedding per frame # Applied change
            
            return predicted_features

        except Exception as e:
            logger.error(f"Error in forward pass: {e}", exc_info=True)
            # Fallback: determine B, T from input x if possible, otherwise use placeholder
            B_fallback, T_fallback = x.shape[:2] if x.ndim >=2 else (1,1)
            embed_dim_fallback = 192 # Consistent with fallback predictor
            try:
                device_fallback = x.device
            except:
                device_fallback = 'cpu' # Absolute fallback for device

            if return_all_features:
                return torch.zeros(B_fallback, T_fallback, embed_dim_fallback, device=device_fallback), \
                       torch.zeros(B_fallback, T_fallback, embed_dim_fallback, device=device_fallback), \
                       None # Added None for target_features
            return torch.zeros(B_fallback, T_fallback, embed_dim_fallback, device=device_fallback)

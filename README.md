# Video Diffusion Model with Latent JEPA

This repository contains an implementation of a vision-language joint embedding predictive architecture (JEPA) for video understanding, optimized specifically for M1 MacBook Air (8GB RAM) systems. The model leverages a frozen ViT encoder and lightweight diffusion decoders to learn spatial-temporal representations from video clips.

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Generating Samples](#generating-samples)
- [M1 Optimization Details](#m1-optimization-details)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## 🌟 Overview

This project implements a hybrid architecture combining:

1. **Frozen ViT Encoder**: Uses a pre-trained Vision Transformer as a feature extractor.
2. **JEPA Training**: Implements masked prediction of video patches for self-supervised learning.
3. **Latent Diffusion**: Generates new video frames by sampling the learned latent space.

The implementation is specifically optimized for Apple Silicon M1 machines with limited memory, using techniques like gradient checkpointing, mixed-precision training, and efficient attention mechanisms.

## 💻 Requirements

- Python 3.10+
- PyTorch 2.1.2
- M1 Mac with MPS support (optimized for 8GB RAM)
- HMDB51 dataset or similar video dataset

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-jepa.git
cd video-jepa
```

2. Create a conda environment:
```bash
conda create -n vjepa python=3.10
conda activate vjepa
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify MPS support:
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## 📁 Project Structure

```
.
├── configs/             # Configuration files
│   └── vjepa_tiny.yaml  # Main configuration
├── data/                # Dataset handling
│   └── dataset.py       # HMDB51 dataset wrapper
├── models/              # Model definitions
│   ├── diffusion_decoder.py  # Latent diffusion decoder
│   └── vjepa_tiny.py    # ViT-based JEPA model
├── utils/               # Utility functions
│   ├── masking.py       # Tube masking for videos
│   ├── memory.py        # Memory tracking utilities
│   └── viz.py           # Visualization tools
├── generate.py          # Generation script
├── train.py             # Training script
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## 🎬 Dataset Preparation

This project is configured to use the HMDB51 dataset, but it can be adapted for any video dataset.

### Using HMDB51

1. Download the HMDB51 dataset from the [official website](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

2. Extract the dataset into a directory structure as follows:
```
data/hmdb51/subset/
├── brush_hair/         # Action category
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── cartwheel/
│   └── ...
└── ...                 # Other action categories
```

3. Set the environment variable to your dataset location:
```bash
export HMDB51_DIR=/path/to/your/hmdb51/subset
```

### Using Custom Datasets

To use a different dataset, modify the `HMDB51Wrapper` class in `data/dataset.py` to match your dataset structure. The expected format is a directory containing subdirectories for each class/category, with video files inside.

## 🚀 Training

To start training with default settings:

```bash
python train.py
```

### Training Options

The training script is configured with sensible defaults for M1 MacBooks with 8GB RAM:

- Batch size: 1 with gradient accumulation of 4 steps
- Learning rate: 5e-5
- Maximum epochs: 30
- Mixed precision (float16)
- MPS acceleration when available

### Monitoring Training

The training script provides rich, colorful output with detailed progress information:

- Real-time loss tracking
- Memory usage statistics
- ETA for training completion
- Progress bars with time estimation

Training checkpoints are saved to the `checkpoints/` directory, with TensorBoard logs in the `logs/` directory.

### Resuming Training

To resume training from a checkpoint:

```bash
python train.py --resume --ckpt_path checkpoints/vjepa-last.ckpt
```

## 🎨 Generating Samples

After training, you can generate new video frames using:

```bash
python generate.py --num_samples 2 --output_dir outputs
```

### Generation Options

- `--num_samples`: Number of samples to generate (default: 2)
- `--output_dir`: Directory to save generated samples (default: 'outputs')
- `--encoder_path`: Path to the trained encoder checkpoint
- `--latent_dim`: Latent dimension size (default: 192)

The generated samples will be saved as PNG images in the specified output directory.

## 🍎 M1 Optimization Details

This implementation includes several optimizations specifically for M1 MacBooks with limited memory:

1. **MPS Acceleration**: Uses Apple's Metal Performance Shaders for GPU acceleration
2. **Memory Management**:
   - Efficient memory tracking and garbage collection
   - Low-memory tensor operations with chunked processing
   - Regular cache clearing to prevent OOM errors
   
3. **Model Optimizations**:
   - Reduced model size with fewer layers and channels
   - Frozen encoder to reduce memory requirements
   - Half-precision (float16) for weights and computations
   - Memory-efficient attention mechanisms
   
4. **Training Efficiency**:
   - Gradient accumulation for effective larger batch sizes
   - Gradient checkpointing to reduce memory usage
   - Lightweight predictor with minimal parameters
   - Batched processing with smaller chunk sizes

## ❓ Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size further in `configs/vjepa_tiny.yaml`
   - Reduce `chunk_size` in model forward passes
   - Try disabling MPS with `export PYTORCH_ENABLE_MPS_FALLBACK=1`

2. **Slow Training**:
   - Ensure MPS acceleration is properly enabled
   - Try adjusting `num_workers` in data loaders
   - Consider enabling memory caching with `cache_mode='memory'` in HMDB51Wrapper

3. **Dataset Loading Errors**:
   - Verify dataset path is correct with `HMDB51_DIR` environment variable
   - Check that video files are compatible formats (mp4, avi, mov)
   - Ensure PyAV is properly installed with `pip install av`

4. **Model Import Errors**:
   - Ensure transformers version matches requirements (4.36.2)
   - Try reinstalling with `pip install -r requirements.txt --force-reinstall`

### Debugging

For advanced debugging, you can increase logging verbosity:

```bash
python train.py --verbose
```

To profile memory usage during training:

```bash
python -m utils.memory
```

## 🛠️ Advanced Usage

### Custom Configurations

You can modify the main configuration file `configs/vjepa_tiny.yaml` to adjust:

- Model architecture (latent dimensions, frozen components)
- Training parameters (learning rate, epochs, etc.)
- Dataset parameters (frame size, clip length)
- Diffusion configuration (timesteps, scheduler)

### Custom Tube Masking

The default implementation uses tube masking for temporal consistency. You can modify `utils/masking.py` to experiment with different masking strategies.

### Using Pre-trained Models

To use a different pre-trained ViT encoder, modify the `pretrained` parameter in `models/vjepa_tiny.py`:

```python
model = TinyVJEPA(pretrained="google/vit-base-patch16-224", freeze_encoder=True)
```

### Multi-GPU Training

While this implementation is optimized for single-GPU M1 systems, you can adapt it for multi-GPU training by modifying the Lightning Trainer:

```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=2,  # Use 2 GPUs
    strategy="ddp",  # Distributed data parallel
    # ... other settings
)
```

## 📚 Citation

If you use this code in your research, please cite our work:

```bibtex
@misc{videojepa2025,
  author = {Your Name},
  title = {Video Diffusion Model with Latent JEPA},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/video-jepa}}
}
```

## 🙏 Acknowledgments

- The ViT implementation is based on HuggingFace's Transformers library
- The diffusion components are adapted from Diffusers library
- Training framework utilizes PyTorch Lightning
- Special thanks to the PyTorch team for MPS support

---

Good luck with your video diffusion adventures! For any questions or issues, please open an issue on GitHub or contact the maintainers.
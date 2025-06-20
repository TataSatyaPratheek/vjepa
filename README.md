An updated `README.md` that aligns with the recent changes in the codebase is provided below. The updates reflect the shift from a generative model to a self-supervised learning and evaluation framework, including a revised project structure, new evaluation scripts, and corrected commands.

***

# Video Joint-Embedding Predictive Architecture (V-JEPA)

This repository contains an implementation of a Video Joint-Embedding Predictive Architecture (V-JEPA) for self-supervised video understanding. The model is optimized to run on M1 MacBook Air systems with 8GB of RAM. It learns spatio-temporal representations from video clips by predicting features in a latent space, using a frozen ViT encoder and a lightweight predictor network.

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [M1 Optimization Details](#m1-optimization-details)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## 🌟 Overview

This project implements a self-supervised learning architecture combining:

1.  **Frozen ViT Encoder**: Uses a pre-trained Vision Transformer as a feature extractor to generate representations for video frames.
2.  **JEPA Training**: Implements a masked prediction task in the latent space, where the model learns to predict features of masked-out portions of a video from the visible context.
3.  **Downstream Evaluation**: The learned representations are evaluated on downstream tasks like video and image classification using linear probing.

The implementation is specifically optimized for Apple Silicon M1 machines with limited memory, using techniques such as gradient checkpointing, mixed-precision training, and efficient memory management.

## 💻 Requirements

-   Python 3.10+
-   PyTorch
-   PyTorch Lightning
-   An M1 Mac with MPS support (optimized for 8GB RAM)
-   HMDB51 dataset or a similar video dataset
-   A full list of dependencies is available in `requirements.txt`.

## 🔧 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/TataSatyaPratheek/vjepa.git
    cd vjepa
    ```

2.  Create a conda environment:
    ```bash
    conda create -n vjepa python=3.10
    conda activate vjepa
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Verify MPS support:
    ```bash
    python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
    ```

## 📁 Project Structure

The project structure has been updated to include evaluation scripts and new utilities.

```
.
├── configs/             # Configuration files
│   └── vjepa_tiny.yaml  # Main configuration
├── data/                # Dataset handling
│   └── dataset.py       # HMDB51 dataset wrapper
├── evals/               # Evaluation scripts
│   ├── image_classification.py
│   └── video_classification.py
├── models/              # Model definitions
│   └── vjepa_tiny.py    # ViT-based JEPA model
├── utils/               # Utility functions
│   ├── mask_collators.py
│   ├── masking.py       # Tube masking for videos
│   ├── memory.py        # Memory tracking utilities
│   ├── schedulers.py    # Learning rate and momentum schedulers
│   └── viz.py           # Visualization tools
├── evaluate.py          # Evaluation script
├── train.py             # Training script
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## 🎬 Dataset Preparation

This project is configured for the HMDB51 dataset but can be adapted for other video datasets.

### Using HMDB51

1.  Download the HMDB51 dataset from the [official website](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

2.  Extract the dataset into a directory. The structure should be as follows:
    ```
    /path/to/your/hmdb51/
    ├── brush_hair/         # Action category
    │   ├── video1.avi
    │   └── ...
    ├── cartwheel/
    │   └── ...
    └── ...                 # Other action categories
    ```

3.  Set the environment variable to your dataset location:
    ```bash
    export HMDB51_DIR=/path/to/your/hmdb51
    ```

### Using Custom Datasets

To use a different dataset, you can modify the `HMDB51Wrapper` class in `data/dataset.py` to match your dataset's structure.

## 🚀 Training

To start training with the default M1-optimized settings:

```bash
python train.py
```

### Training Options

The training script is configured with defaults suitable for M1 MacBooks with 8GB RAM:
-   Batch size: 1 with gradient accumulation of 4 steps
-   Learning rate: 5e-5
-   Maximum epochs: 30
-   Mixed-precision (float16) training
-   MPS acceleration when available

### Monitoring Training

The script provides a rich, colorful console output with detailed progress, including real-time loss, memory usage, and ETA for completion. Training checkpoints are saved to the `checkpoints/` directory, and TensorBoard logs are stored in the `logs/` directory.

## 🎨 Evaluation

After training, you can evaluate the learned representations on downstream classification tasks using the `evaluate.py` script. This script performs linear probing on both video (HMDB51) and image (CIFAR-10) datasets.

To run the evaluation with a trained checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/vjepa-last.ckpt
```

### Evaluation Options

-   `--checkpoint`: Path to the trained model checkpoint.
-   `--data_dir`: Path to the video dataset (defaults to `data/hmdb51/subset`).
-   `--batch_size`: Batch size for evaluation (default: 16).
-   `--device`: Device to use (`auto`, `cpu`, `cuda`, `mps`).

The script will output the classification accuracy for both tasks.

## 🍎 M1 Optimization Details

This implementation includes several optimizations for M1 MacBooks with limited memory:
1.  **MPS Acceleration**: Utilizes Apple's Metal Performance Shaders for GPU acceleration.
2.  **Memory Management**: Implements efficient memory tracking, periodic garbage collection, and cache clearing to prevent out-of-memory errors.
3.  **Model Optimizations**: Employs a frozen encoder, a reduced model size, half-precision (float16) computations, and memory-efficient attention mechanisms.
4.  **Training Efficiency**: Uses gradient accumulation to simulate larger batch sizes, gradient checkpointing to save memory, and a lightweight predictor network.

## ❓ Troubleshooting

### Common Issues

1.  **Out of Memory Errors**:
    -   Reduce the batch size in `configs/vjepa_tiny.yaml`.
    -   Try disabling MPS with `export PYTORCH_ENABLE_MPS_FALLBACK=1`.
2.  **Slow Training**:
    -   Ensure MPS acceleration is enabled.
    -   Adjust `num_workers` in the data loaders in `train.py`.
    -   Enable memory caching by setting `cache_mode='memory'` in `HMDB51Wrapper` if you have sufficient RAM.
3.  **Dataset Loading Errors**:
    -   Verify the `HMDB51_DIR` environment variable points to the correct path.
    -   Check that video files are in compatible formats (e.g., .mp4, .avi).
    -   Ensure PyAV is installed correctly (`pip install av`).

## 🛠️ Advanced Usage

### Custom Configurations

You can modify the main configuration file `configs/vjepa_tiny.yaml` to adjust model architecture, training parameters, and dataset settings.

### Custom Tube Masking

The project uses a tube masking strategy for temporal consistency. You can experiment with different masking approaches by modifying `utils/masking.py` and `utils/mask_collators.py`.

### Using Pre-trained Models

To use a different pre-trained ViT encoder, modify the `pretrained` parameter in `models/vjepa_tiny.py`:
```python
model = TinyVJEPA(pretrained="google/vit-base-patch16-224", freeze_encoder=True)
```

## 📚 Citation

If you use this code in your research, please cite the repository:

```bibtex
@misc{videojepa2025,
  author = {Satya Pratheek Tata},
  title = {V-JEPA: A Video Joint-Embedding Predictive Architecture Implementation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TataSatyaPratheek/vjepa}}
}
```

## 🙏 Acknowledgments

-   The ViT implementation is based on Hugging Face's `transformers` library.
-   The training framework utilizes PyTorch Lightning.
-   Special thanks to the PyTorch team for their work on MPS support.

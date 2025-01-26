# Diffusion Models with Keras

This repository implements Denoising Diffusion Implicit Models (DDIM) using Keras. The implementation includes a complete training and inference pipeline for generating images using the CIFAR-10 dataset.

## Overview

The project implements:
- DDIM architecture with configurable U-Net backbone
- Forward and reverse diffusion processes
- Training pipeline with EMA model averaging
- Inference pipeline for image generation
- Visualization utilities

## Requirements

- TensorFlow 
- Keras
- NumPy
- Matplotlib

## Setup

Activate the TensorFlow Metal virtual environment (for Mac GPU support):

```bash
source ~/.tf-metal/bin/activate
```

## Usage
### Training
Train a new DDIM model on CIFAR-10:

```bash
python training.py
```

This will: 
- Load and preprocess the CIFAR-10 dataset
- Train the diffusion model for the configured number of epochs
- Save model checkpoints and generated samples during training

### Inference
Generate new images using a trained model:

```bash
python inference.py
```


## Project Structure
`architectures.py` - U-Net backbone implementation
`constants.py` - Model and training configuration
`ddim.py` - Core DDIM model implementation
`training.py` - Training pipeline
`inference.py` - Image generation pipeline
`utils.py` - Visualization and helper utilities

## Configuration
Key parameters can be configured in `constants.py`:
- Image dimensions and channels
- Model architecture (widths, block depth)
- Training parameters (batch size, learning rate, etc.)
- Diffusion process parameters
- Visualization settings

## Model Checkpoints
Trained model checkpoints are saved in the `models` directory with timestamps.
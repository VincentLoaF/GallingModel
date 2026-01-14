# Complete Training Guide

## Overview

Three models available, each with dedicated training script:

| Model | Parameters | Script | Complexity |
|-------|-----------|--------|------------|
| **Feedforward PINN** | 19,271 | `train_feedforward.py` | Baseline |
| **CNN-Hybrid PINN** | 7,479 | `train_cnn.py` | 61% fewer params |
| **Physics-Only** | 8 | `train_physics_only.py` | Minimal |

## Quick Start

### Train Feedforward Model (Recommended First)

```bash
python scripts/train_feedforward.py
```

**What happens:**
1. Loads 536 cycles from 3 temperatures
2. Stage 1: Pre-trains neural network (50 epochs)
3. Stage 2: Jointly optimizes NN + physics (100 epochs)
4. Saves to `results/feedforward/best_stage2.pth`

**Time:** ~10-15 min (GPU) or ~30-45 min (CPU)

### Train CNN Model

```bash
python scripts/train_cnn.py
```

**Differences from feedforward:**
- Uses 1D convolutional layers for local temporal patterns
- 61% fewer parameters (7,479 vs 19,271)
- Same 2-stage training process
- Saves to `results/cnn/best_stage2.pth`

### Train Physics-Only Model

```bash
python scripts/train_physics_only.py
```

**Differences:**
- No neural network, only 8 physics parameters
- Single-stage training (no Stage 1)
- Faster training (~5-10 min)
- Saves to `results/physics_only/best_model.pth`
- Fully interpretable mechanistic model

## Training Outputs

### Directory Structure

After training feedforward:
```
results/feedforward/
├── best_stage1.pth              # Best Stage 1 model
├── best_stage2.pth              # ⭐ Final model (use this!)
├── fitted_parameters.json       # Physics parameters
├── checkpoint_stage2_epoch10.pth  # Training checkpoints
├── checkpoint_stage2_epoch20.pth
└── ...
```

### What's in Each File

**best_stage2.pth** (main model):
```python
{
    'model_state_dict': {...},      # Model weights
    'epoch': 100,                   # Training epoch
    'val_loss': 0.0043,            # Validation loss
}
```

**fitted_parameters.json** (physics params):
```json
{
    "k0": 0.873,      // Attachment rate
    "T0": 24.874,     // Temperature scale
    "kw": 0.203,      // Wear coefficient
    "M0": 10.0,       // Critical mass
    "Tc": 45.0,       // Critical temp scale
    "alpha": 0.2      // Detachment fraction
}
```

## Training Monitoring

### Console Output

The scripts print progress every epoch:

```
Epoch 15/100 | Train Loss: 0.0234 | Val Loss: 0.0189 | LR: 1.00e-04

--- Physics Parameters (Epoch 20) ---
  k0        = 0.873
  T0        = 24.874
  kw        = 0.203
  M0        = 10.000
  Tc        = 45.000
  alpha     = 0.200
```

### What to Look For

✅ **Good signs:**
- Validation loss decreasing
- Physics parameters converging
- No huge spikes in loss

⚠️ **Warning signs:**
- Val loss increasing → May be overfitting
- Loss not decreasing → Learning rate too high/low
- NaN values → Numerical instability

## Customization

### Change Hyperparameters

Edit config files before training:

**config/feedforward.yaml:**
```yaml
training:
  stage1:
    n_epochs: 50                # Increase for better pre-training
    learning_rate: 0.001        # Higher = faster but less stable
    batch_size: 32              # Reduce if CUDA out of memory

  stage2:
    n_epochs: 100               # Increase for better convergence
    learning_rate_physics: 0.001  # Physics params learn faster
    loss_weights:
      w_incycle: 1.0            # In-cycle COF importance
      w_physics: 0.5            # Physics constraint importance
```

### Change Model Architecture

**config/feedforward.yaml:**
```yaml
model:
  hidden_dims: [64, 128, 64, 32]  # Layer sizes
  dropout: 0.1                     # Dropout rate (0-0.5)
```

**config/cnn.yaml:**
```yaml
model:
  conv_channels: [16, 32]    # Conv output channels
  kernel_sizes: [5, 5]       # Temporal window sizes
  fc_hidden: [64, 32]        # Feedforward layers after conv
```

### Use Subset of Data

Modify training script (e.g., train_feedforward.py line 31):

```python
# Use only specific temperatures
dataset = data_loader.create_pytorch_dataset([165, 170])  # Skip 167.5°C

# Or limit number of cycles
dataset = data_loader.create_pytorch_dataset([165, 167.5, 170])
dataset = dataset[:100]  # Use first 100 cycles only
```

## After Training

### 1. Visualize Predictions

```bash
python scripts/plot_model_predictions.py
```

Generates: `results/feedforward/predictions_vs_observed.png`

### 2. Compare Models

```bash
python experiments/compare_models.py
```

Trains all 3 models and compares:
- Prediction accuracy (R², RMSE)
- Model complexity (parameters)
- Training time
- Physics parameters

### 3. Use Trained Model

```python
import torch
import sys
sys.path.append('src')
from models.pinn_feedforward import GallingPINN

# Load model
model = GallingPINN()
checkpoint = torch.load('results/feedforward/best_stage2.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
output = model.forward_multi_cycle(
    T=165,              # Temperature
    features_list=[...], # Feature tensors
    M_init=0.0          # Initial mass
)

cof = output['cof_predicted']      # In-cycle COF predictions
mass = output['M_history']         # Transfer layer mass evolution
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config file:
```yaml
training:
  stage1:
    batch_size: 16  # Default is 32
```

### Training Too Slow

- Verify GPU is being used (check "Device: cuda" in output)
- Reduce number of epochs temporarily for testing
- Use smaller model architecture

### Model Not Converging

**Stage 1 not converging:**
- Increase Stage 1 epochs (50 → 100)
- Reduce learning rate (0.001 → 0.0005)
- Check data quality

**Stage 2 not converging:**
- Reduce physics learning rate
- Increase `w_physics` weight
- Better Stage 1 pre-training first

### NaN Loss

- Reduce learning rates (both NN and physics)
- Add gradient clipping (already enabled in configs)
- Check for data issues (infinite values, NaNs)

## Advanced Usage

### Resume Training

Modify training script to load checkpoint:

```python
# Load checkpoint
checkpoint = torch.load('results/feedforward/checkpoint_stage2_epoch50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Continue training from epoch 51
```

### Transfer Learning

Use pre-trained weights from one temperature to another:

```python
# Train on 165°C first
dataset_165 = loader.create_pytorch_dataset([165])
trainer.stage1_pretrain_nn(dataset_165)

# Fine-tune on 170°C
dataset_170 = loader.create_pytorch_dataset([170])
trainer.stage2_joint_training(dataset_170)
```

### Custom Physics Initialization

Modify config before training:

```yaml
model:
  physics_init:
    k0: 0.5      # Different starting values
    T0: 30.0
    kw: 0.15
    # ... etc
```

## Model Comparison Tips

### When to Use Each Model

**Feedforward PINN:**
- Best overall accuracy
- Most parameters to learn
- Good for understanding full temporal patterns

**CNN-Hybrid:**
- Good accuracy with fewer parameters
- Captures local temporal patterns efficiently
- Best for deployment (smaller model size)

**Physics-Only:**
- Baseline for comparison
- Fully interpretable
- Shows value of data-driven components

### Comparison Metrics

Run all 3 and compare:
```bash
python experiments/compare_models.py
```

Look at:
- **R²**: Higher is better (>0.95 is excellent)
- **RMSE**: Lower is better (<0.05 is good)
- **Parameters**: Fewer is better (for deployment)
- **Training time**: Faster is better

## Questions?

- See `README.md` for project overview
- See `QUICK_START.md` for quick reference
- See `SRC_REFACTORING.md` for code structure
- Check config files for all hyperparameters

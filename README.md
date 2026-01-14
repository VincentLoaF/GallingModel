# Galling Prediction using Physics-Informed Neural Networks (PINN)

A hybrid machine learning approach combining data-driven neural networks with physics-based mechanistic models to predict tribological galling in hot stamping processes.

## Project Overview

This project develops and compares three modeling approaches for predicting coefficient of friction (COF) and galling behavior during aluminum hot stamping:

1. **Feedforward PINN**: Neural network learns temporal patterns + physics equations
2. **CNN-Hybrid PINN**: 1D CNN captures local patterns + physics equations
3. **Physics-Only**: Pure mechanistic model without neural network

**Key Results**: R² = 0.96, RMSE = 0.043 on 536 cycles across 3 temperatures

## Project Structure

```
GallingModel/
├── data/                          # Experimental data
│   ├── 165/                      # 165°C experiments
│   ├── 167.5/                    # 167.5°C experiments
│   ├── 170/                      # 170°C experiments
│   └── 25/                       # Room temperature data
│
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── pinn_feedforward.py   # Feedforward PINN (19,271 params)
│   │   ├── pinn_cnn.py           # CNN-Hybrid PINN (7,479 params)
│   │   └── physics_model.py      # Pure physics, no NN (8 params)
│   ├── trainers/                 # Training logic
│   │   ├── trainer_feedforward.py  # 2-stage trainer (FF & CNN)
│   │   └── trainer_physics_only.py # Single-stage trainer
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   └── visualization.py          # Plotting utilities
│
├── config/                       # Configuration files
│   ├── feedforward.yaml          # Feedforward PINN config
│   ├── cnn.yaml                  # CNN-Hybrid config
│   └── physics_only.yaml         # Physics-only config
│
├── experiments/                  # Experiment scripts
│   └── compare_models.py         # Three-way model comparison
│
├── scripts/                      # Training & utility scripts
│   ├── train_feedforward.py      # Train feedforward model
│   ├── train_cnn.py              # Train CNN model
│   ├── train_physics_only.py     # Train physics-only model
│   ├── plot_model_predictions.py # Visualization
│   └── plot_comparison.py        # Comparison plots
│
├── results/                      # Training outputs
│   ├── feedforward/              # Feedforward model results
│   ├── cnn/                      # CNN model results
│   ├── physics_only/             # Physics-only results
│   ├── comparison_ff_vs_cnn.json # Two-way comparison
│   └── comparison_all_three.json # Three-way comparison
│
└── docs/                         # Documentation archive
```

## Quick Start

### 1. Train Individual Models

Each model has a dedicated training script:

```bash
# Train feedforward PINN (19,271 parameters)
python scripts/train_feedforward.py

# Train CNN-hybrid PINN (7,479 parameters, 61% fewer)
python scripts/train_cnn.py

# Train physics-only model (8 parameters, minimal)
python scripts/train_physics_only.py
```

**What each script does:**
- Loads data from all three temperatures (165°C, 167.5°C, 170°C)
- Trains the model (2-stage for NN models, single-stage for physics-only)
- Saves best model, training history, and parameters
- Prints progress and final results

**Expected time**: ~10-15 minutes per model (GPU) or ~30-45 minutes (CPU)

### 2. Compare All Three Models

```bash
# Run comprehensive comparison
python experiments/compare_models.py
```

This will:
- Train all three models with identical data
- Evaluate prediction accuracy (R², RMSE)
- Compare model complexity and training time
- Save results to `results/comparison_all_three.json`

### 3. Visualize Results

```bash
# Plot predictions for feedforward model
python scripts/plot_model_predictions.py

# Generate comparison plots
python scripts/plot_comparison.py
```

## Model Architectures

### Feedforward PINN (Baseline)
- **Parameters**: 19,271
- **Architecture**: Input(8) → FC[64, 128, 64, 32] → Output(1)
- **Physics**: Arrhenius attachment + friction-dependent wear
- **Training**: 2-stage (NN pre-train + joint optimization)

### CNN-Hybrid PINN
- **Parameters**: 7,479 (61% reduction)
- **Architecture**:
  - Conv1D: 8 → 16 → 32 channels (kernel=5)
  - FC: 32 → 64 → 32 → 1
- **Advantage**: Captures local temporal patterns with fewer parameters
- **Training**: 2-stage (same as feedforward)

### Physics-Only
- **Parameters**: 8 (physics parameters only)
- **Model**: μ = μ_base + μ_slope × M
- **Advantage**: Fully interpretable, minimal complexity
- **Training**: Direct physics parameter optimization

## Key Features

### Two-Stage Training
1. **Stage 1**: Pre-train neural network on in-cycle COF data (supervised)
2. **Stage 2**: Jointly optimize NN + physics parameters (physics-informed)

### Physics Integration
All models use identical mass balance equations:
```
M(n+1) = M(n) + Q_attach(T) - Q_wear(M, μ)

where:
  Q_attach = k0 * exp((T - T_ref) / T0)  # Arrhenius temperature dependence
  Q_wear = kw * M * μ                    # Friction-dependent wear

  If M ≥ M_crit: detachment occurs, M → α * M
```

### Temperature Reference
All models use **T_ref = 165°C** (lowest experimental temperature) for improved parameter fitting stability.

## Data

- **125 Hz in-cycle measurements**: Force, position, velocity, COF
- **Total cycles**: 536 across 3 temperatures
  - 165°C: 114 cycles (transient galling)
  - 167.5°C: 280 cycles (oscillatory behavior)
  - 170°C: 142 cycles (permanent galling)
- **Total timesteps**: 155,522

## Results

### Model Comparison

| Model | Parameters | R² | RMSE (cycle-avg) | Training Time |
|-------|-----------|-----|------------------|---------------|
| **Feedforward PINN** | 19,271 | 0.96 | 0.043 | ~baseline |
| **CNN-Hybrid PINN** | 7,479 | ~0.96 | ~0.043 | ~1.0x |
| **Physics-Only** | 8 | TBD | TBD | ~0.3x |

*CNN achieves similar performance with 61% fewer parameters through weight sharing*

### Key Insights
1. **Neural network adds value**: Captures complex temporal patterns not in simplified physics
2. **CNN efficiency**: Local pattern detection with parameter sharing reduces model size
3. **Physics constraints**: Ensure predictions follow mechanistic principles

## Configuration

All models share common training settings (Stage 2 example):
```yaml
training:
  stage2:
    n_epochs: 100
    learning_rate_physics: 0.001
    loss_weights:
      w_incycle: 1.0        # In-cycle COF matching
      w_physics: 0.5        # Physics constraint
      w_regularization: 0.1 # Parameter regularization
```

Modify `config/*.yaml` to adjust hyperparameters.

## Training Guide

### Training Individual Models

#### Feedforward PINN (Recommended Starting Point)
```bash
python scripts/train_feedforward.py
```

**What it does:**
- Loads 536 cycles from 3 temperatures
- Stage 1: Pre-trains neural network (50 epochs)
- Stage 2: Jointly optimizes NN + physics (100 epochs)
- Saves best model, plots, and training history

**Outputs:**
- `results/feedforward/best_stage1.pth` - Best Stage 1 checkpoint
- `results/feedforward/best_stage2.pth` - Best Stage 2 checkpoint (use this!)
- `results/feedforward/fitted_parameters.json` - Final physics parameters
- `results/feedforward/checkpoint_stage2_epoch*.pth` - Training checkpoints

**Monitor training:**
The script prints progress every epoch with train/val loss and physics parameters.

#### CNN-Hybrid PINN
```python
import sys; sys.path.append('src')
from models.pinn_cnn import GallingPINN_CNN
from data_loader import HighFrequencyDataLoader
from train import PINNTrainer
import yaml

# Load config and data
config = yaml.safe_load(open('config/cnn.yaml'))
loader = HighFrequencyDataLoader()
data = loader.create_pytorch_dataset([165, 167.5, 170])

# Create and train model
model = GallingPINN_CNN()
trainer = PINNTrainer(model, config)
trainer.stage1_pretrain_nn(data)
trainer.stage2_joint_training(data)
```

#### Pure Physics Model (No Neural Network)
```python
import sys; sys.path.append('src')
from models.physics_model import GallingPhysicsModel
from data_preprocessing import HighFrequencyDataLoader
from trainers.trainer_physics_only import PhysicsOnlyTrainer
import yaml

# Load config and data
config = yaml.safe_load(open('config/physics_only.yaml'))
loader = HighFrequencyDataLoader()
data = loader.create_pytorch_dataset([165, 167.5, 170])

# Create and train model (single stage, faster)
model = GallingPhysicsModel()
trainer = PhysicsOnlyTrainer(model, config)
trainer.train(data)  # No Stage 1, goes straight to physics optimization
```

### Customizing Training

Edit configuration files to adjust hyperparameters:

**Key settings in `config/feedforward.yaml`:**
```yaml
training:
  stage1:
    n_epochs: 50              # Stage 1 epochs
    learning_rate: 0.001      # Learning rate

  stage2:
    n_epochs: 100             # Stage 2 epochs
    learning_rate_nn: 0.0001  # NN learning rate
    learning_rate_physics: 0.001  # Physics LR (can be higher)
    loss_weights:
      w_incycle: 1.0          # In-cycle COF importance
      w_physics: 0.5          # Physics constraint importance
```

### Loading Trained Models

```python
import torch
from models.pinn_feedforward import GallingPINN

# Load model
model = GallingPINN()
checkpoint = torch.load('results/feedforward/best_stage2.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for prediction
# output = model.forward_multi_cycle(temp, features_list)
```

## Requirements

```
torch>=1.13.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
pyyaml>=5.4.0
tqdm>=4.62.0
```

## Citation

If you use this work, please cite:
```bibtex
@misc{galling_pinn_2026,
  title={Physics-Informed Neural Networks for Tribological Galling Prediction},
  author={Your Name},
  year={2026}
}
```

## License

MIT License

## Contact

For questions or collaborations, please contact [your email].

---

**Last Updated**: 2026-01-13

# Quick Start Guide

## Training Feedforward PINN (Simplest Method)

```bash
python scripts/train_feedforward.py
```

That's it! This single command will:
1. Load all experimental data (536 cycles)
2. Train the model (Stage 1 + Stage 2)
3. Save results to `results/feedforward/`

**Expected time:** ~10-15 minutes

## What Gets Saved

After training completes:

```
results/feedforward/
├── best_stage1.pth              # Best Stage 1 model
├── best_stage2.pth              # ⭐ Best final model (use this!)
├── fitted_parameters.json       # Final physics parameters
└── checkpoint_stage2_epoch*.pth # Training checkpoints (every 10 epochs)
```

## View Results

### 1. Plot Predictions
```bash
python scripts/plot_model_predictions.py
```

Generates: `results/feedforward/predictions_vs_observed.png`

Shows:
- Predicted vs observed COF for each temperature
- R² and RMSE metrics
- Cycle-by-cycle comparison

### 2. Compare with Other Models

```bash
python experiments/compare_models.py
```

Trains all 3 models and compares:
- Feedforward (19,271 params)
- CNN-Hybrid (7,479 params)
- Physics-Only (8 params)

Saves: `results/comparison_all_three.json`

## Training Details

### Two-Stage Training

**Stage 1: Neural Network Pre-training**
- Objective: Learn to predict in-cycle COF from features
- Loss: Mean squared error between predicted and observed COF
- Epochs: 50 (default)
- Physics parameters: Frozen (not optimized)

**Stage 2: Joint Optimization**
- Objective: Optimize both NN and physics to match cycle-averaged COF
- Loss: Weighted sum of in-cycle loss + physics constraint + regularization
- Epochs: 100 (default)
- Physics parameters: Learnable (jointly optimized with NN)

### Monitor Progress

The script prints updates every epoch:

```
Epoch 1/100 | Train Loss: 0.0234 | Val Loss: 0.0189 | LR: 1.00e-04

--- Physics Parameters (Epoch 10) ---
  k0        = 0.873
  T0        = 24.874
  kw        = 0.203
  ...
```

Look for:
- ✅ Decreasing validation loss
- ✅ Physics parameters converging
- ⚠️ If val loss increases: model may be overfitting

## Customization

### Change Hyperparameters

Edit `config/feedforward.yaml`:

```yaml
training:
  stage1:
    n_epochs: 50                # Increase for better pre-training
    learning_rate: 0.001        # Higher = faster but less stable

  stage2:
    n_epochs: 100               # Increase for better convergence
    learning_rate_physics: 0.001  # Physics params LR
```

### Use Different Temperatures

Modify the script to use specific temperatures:

```python
# In train_feedforward.py, line ~31
dataset = data_loader.create_pytorch_dataset([165, 170])  # Only 165 and 170°C
```

### Change Model Architecture

Edit `config/feedforward.yaml`:

```yaml
model:
  hidden_dims: [64, 128, 64, 32]  # Change layer sizes
  dropout: 0.1                     # Dropout probability
```

## Common Issues

### CUDA Out of Memory
Reduce batch size in `config/feedforward.yaml`:
```yaml
training:
  stage1:
    batch_size: 16  # Default is 32
```

### Training Too Slow
- Use GPU if available (automatic)
- Reduce number of epochs
- Use smaller model: `hidden_dims: [32, 64, 32]`

### Model Not Converging
- Increase Stage 1 epochs for better pre-training
- Reduce learning rates
- Check data quality

## Next Steps

After training:

1. **Visualize**: `python scripts/plot_model_predictions.py`
2. **Compare**: `python experiments/compare_models.py`
3. **Use model**: Load `best_stage2.pth` for predictions
4. **Experiment**: Try CNN or physics-only models
5. **Tune**: Adjust hyperparameters for better performance

## Advanced Usage

### Load and Use Trained Model

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

# Make predictions for a temperature
output = model.forward_multi_cycle(
    T=165,                    # Temperature in °C
    features_list=[...],      # List of feature tensors (one per cycle)
    M_init=0.0               # Initial transfer layer mass
)

# Extract predictions
cof_predicted = output['cof_predicted']           # In-cycle COF
mu_mean = output['mu_mean_history']              # Cycle-averaged COF
mass_history = output['M_history']               # Transfer layer mass
```

### View Physics Parameters

```python
import json

# Load fitted parameters
with open('results/feedforward/fitted_parameters.json') as f:
    params = json.load(f)

print(params)
# {'k0': 0.873, 'T0': 24.874, 'kw': 0.203, ...}
```

## Questions?

- See full documentation: `README.md`
- Check model details: `docs/` folder
- Compare models: Run `experiments/compare_models.py`

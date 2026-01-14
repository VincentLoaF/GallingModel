# Galling Phenomenon PINN Model

Physics-Informed Neural Network (PINN) for predicting tribological galling in hot stamping processes.

## Overview

This project implements a hybrid physics-ML model that combines:
1. **Physics-based cycle evolution**: Mass balance model for aluminum transfer layer M(n)
2. **Data-driven within-cycle prediction**: Neural network learns COF(x,t) from 125Hz experimental data

### Key Innovation

- Physics model handles **between-cycle** dynamics (attachment, wear, detachment)
- Neural network handles **within-cycle** dynamics (spatial variation, temporal evolution)
- Unified through state variable M(n) that connects cycles

## Project Structure

```
GallingModel/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── pinn_model.py           # Hybrid PINN architecture
│   └── train.py                # Training loops (Stage 1 & 2)
├── notebooks/
│   └── 01_data_exploration.ipynb   # Data exploration
├── config/
│   └── config.yaml             # Hyperparameters and settings
├── results/
│   ├── models/                 # Saved model checkpoints
│   ├── plots/                  # Validation plots
│   └── logs/                   # Training logs
├── main.py                     # CLI interface
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── PINN_Implementation_Plan.md # Detailed implementation plan
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

Expected data structure:
```
GallingModel/
├── 165/
│   ├── data/
│   │   ├── 000001_*.csv
│   │   ├── 000002_*.csv
│   │   └── ...
│   ├── mean.txt
│   └── std.txt
├── 167.5/
│   ├── data/
│   ├── mean.txt
│   └── std.txt
└── 170/
    ├── data/
    ├── mean.txt
    └── std.txt
```

### CSV Format

Each CSV file contains 125Hz in-cycle data with columns:
- `timestamp`, `force_x`, `force_y`, `force_z`
- `position_x`, `position_y`, `position_z`
- `velocity_x`, `cof`, `sliding_distance`

### Preprocessing

- **Filtering**: Rows with `sliding_distance < 0.1mm` are removed (initialization artifacts)
- **Feature engineering**: `cycle_phase` [0,1] added to mark progress through slide

## Usage

### Quick Start

Train both stages with default configuration:
```bash
python main.py --stage both --config config/config.yaml
```

### Stage-by-Stage Training

**Stage 1**: Pre-train neural network (supervised learning)
```bash
python main.py --stage 1 --config config/config.yaml
```

**Stage 2**: Joint physics-informed training
```bash
python main.py --stage 2 --config config/config.yaml
```

### Advanced Options

```bash
# Use specific device
python main.py --stage both --config config/config.yaml --device cuda

# Override epochs
python main.py --stage 1 --config config/config.yaml --epochs 100

# Resume from checkpoint
python main.py --stage 2 --load-checkpoint results/models/best_stage1.pth
```

## Model Architecture

### Neural Network Component

- **Input**: 8 features per timestep
  1. M_normalized (transfer layer mass)
  2. T_normalized (temperature)
  3. sliding_distance_norm
  4. velocity_x
  5. force_x, force_y, force_z
  6. cycle_phase

- **Architecture**: [64, 128, 64, 32] hidden layers with Tanh activation
- **Output**: COF prediction scaled to [0, 1.3]

### Physics Component

Mass balance model with 6 learnable parameters:
1. `k0`: Baseline attachment rate
2. `T0`: Attachment temperature sensitivity (°C)
3. `kw`: Wear rate constant
4. `M0`: Baseline critical mass
5. `Tc`: Critical mass temperature sensitivity (°C)
6. `alpha`: Retention fraction after detachment

**Update equations**:
```
Q_attach = k0 * exp((T - 200) / T0)
Q_wear = kw * M * μ_mean
M(n+1) = M(n) + Q_attach - Q_wear

if M(n+1) >= M_crit:
    M(n+1) = alpha * M(n+1)  # Partial detachment
```

## Training Strategy

### Stage 1: NN Pre-training (Supervised)
- Freeze physics parameters
- Minimize MSE between predicted and observed COF(t)
- Initialize M(n) using mean COF from mean.txt
- **Typical result**: Train loss ~0.01-0.05

### Stage 2: Joint Training (Physics-Informed)
- Unfreeze physics parameters
- Combined loss:
  ```
  L_total = w1 * L_incycle + w2 * L_physics + w3 * L_regularization

  L_incycle: MSE(COF_predicted, COF_observed)
  L_physics: MSE(μ_mean_predicted, μ_mean_observed)
  L_regularization: Physics parameter constraints
  ```
- **Typical result**: Total loss ~0.02-0.05

## Configuration

Edit `config/config.yaml` to customize:

- **Data settings**: Temperatures, filtering threshold
- **Model architecture**: Hidden dimensions, dropout
- **Training**: Epochs, batch size, learning rates, loss weights
- **Validation**: Train/val split, early stopping
- **Output**: Save directories, logging frequency

## Results

After training, results are saved to `results/`:

- `models/best_stage1.pth`: Best Stage 1 model checkpoint
- `models/best_stage2.pth`: Best Stage 2 model checkpoint
- `models/fitted_parameters.json`: Final physics parameters
- `logs/training_history.json`: Loss curves and metrics

## Expected Performance

### Success Criteria

**Minimum (Model is Valid)**:
- In-cycle RMSE < 0.10
- Cycle-averaged RMSE < 0.15
- Physics parameters within expected ranges

**Good (Quantitatively Accurate)**:
- In-cycle RMSE < 0.05
- Cycle-averaged RMSE < 0.10
- M(n) evolution shows realistic accumulation/detachment

**Excellent (Publication-Quality)**:
- In-cycle RMSE < 0.03
- Cycle-averaged RMSE < 0.05
- Physics parameters physically interpretable (T0 ~ 20-30°C, Tc ~ 40-60°C)

## Data Exploration

Launch Jupyter notebook for interactive exploration:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Generates visualizations:
- COF profiles per cycle
- Cycle-averaged evolution
- Force and velocity correlations
- Galling event frequency
- Preprocessing validation

## Testing

Test individual components:

```bash
# Test data loader
python src/data_loader.py

# Test PINN model
python src/pinn_model.py
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config/config.yaml`
- Use `--device cpu` for CPU-only training

### Slow Training
- Enable GPU with `--device cuda`
- Reduce number of cycles for testing
- Decrease hidden layer dimensions

### Physics Parameters Diverging
- Increase regularization weight `w_regularization`
- Decrease learning rate for physics parameters
- Check gradient clipping is enabled

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

## License

[Your license here]

## Contact

For questions or issues, please contact [your email] or open an issue on GitHub.

## References

See `PINN_Implementation_Plan.md` for detailed technical documentation.

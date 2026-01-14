# Plotting Guide

## Overview

Two main plotting scripts for visualizing model results:

1. **plot_model_predictions.py** - Visualize individual model predictions
2. **plot_comparison.py** - Compare multiple models side-by-side

---

## 1. Plot Model Predictions

### Purpose
Visualize how well a single trained model predicts cycle-averaged COF compared to observed data.

### Usage

```bash
# Plot feedforward model predictions
python scripts/plot_model_predictions.py --model feedforward

# Plot CNN model predictions
python scripts/plot_model_predictions.py --model cnn

# Plot pure physics model predictions
python scripts/plot_model_predictions.py --model physics
```

### Advanced Options

```bash
# Use specific checkpoint
python scripts/plot_model_predictions.py --model feedforward \
    --checkpoint results/feedforward/checkpoint_stage2_epoch50.pth

# Custom output path
python scripts/plot_model_predictions.py --model cnn \
    --output my_custom_plot.png

# Custom data path
python scripts/plot_model_predictions.py --model feedforward \
    --data-path /path/to/data
```

### What Gets Plotted

**4-panel figure:**
1. **165°C predictions** - Cycle-by-cycle comparison
2. **167.5°C predictions** - Cycle-by-cycle comparison
3. **170°C predictions** - Cycle-by-cycle comparison
4. **All temperatures** - Predicted vs Observed scatter plot

**Metrics shown:**
- R² (coefficient of determination) per temperature
- RMSE (root mean squared error) per temperature
- Overall R² and RMSE

**Output locations:**
- Feedforward: `results/feedforward/predictions_vs_observed.png`
- CNN: `results/cnn/predictions_vs_observed.png`
- Physics: `results/physics/predictions_vs_observed.png`

### Example Output

```
Device: cuda

Loading data from /root/Documents/GallingModel/data...
✓ Data loaded
Loading feedforward model from results/feedforward/best_stage2.pth...
✓ Model loaded successfully
  Parameters: 19,271

Generating predictions...
  Processing 165°C...
  Processing 167.5°C...
  Processing 170°C...

Creating plot...

✓ Predictions plot saved to: results/feedforward/predictions_vs_observed.png

============================================================
PREDICTION SUMMARY - Feedforward PINN
============================================================
Overall R²:   0.9624
Overall RMSE: 0.0432

Per-temperature R²:
  165°C: 0.9712
  167.5°C: 0.9558
  170°C: 0.9703
============================================================
```

---

## 2. Plot Model Comparison

### Purpose
Create comprehensive comparison visualizations after running the model comparison experiment.

### Prerequisites

**Must run comparison experiment first:**
```bash
python experiments/compare_models.py
```

This creates: `results/comparison_all_three.json`

### Usage

```bash
# Default: Compare all three models
python scripts/plot_comparison.py

# Use custom results file
python scripts/plot_comparison.py --results results/comparison_all_three.json

# Skip physics parameters plot
python scripts/plot_comparison.py --no-params
```

### What Gets Plotted

**Main comparison plot (6 panels):**
1. **Parameter count** - Model complexity comparison (log scale)
2. **Training time** - Computational efficiency
3. **R² scores** - Overall fit quality
4. **RMSE values** - Prediction accuracy
5. **Per-temperature R²** - Performance across temperatures
6. **Per-temperature RMSE** - Accuracy across temperatures

**Physics parameters plot (6 panels):**
- Comparison of fitted physics parameters (k0, T0, kw, M0, Tc, alpha)
- Shows how each model learned different parameter values

**Output locations:**
- Main comparison: `results/comparison_all_three.png`
- Physics params: `results/physics_parameters_comparison.png`

### Text Summary

The script also prints a comprehensive text summary:

```
================================================================================
MODEL COMPARISON SUMMARY
================================================================================

1. MODEL COMPLEXITY:
   Feedforward PINN    :   19,271 parameters
   CNN-Hybrid PINN     :    7,479 parameters
   Pure Physics        :        8 parameters

2. TRAINING TIME:
   Feedforward PINN    : 23.4min (1402.3s)
   CNN-Hybrid PINN     : 18.7min (1123.5s)
   Pure Physics        : 3.2min (192.1s)

3. PREDICTION ACCURACY:
   Model                        R²       RMSE
   -------------------- ---------- ----------
   Feedforward PINN         0.9624     0.0432
   CNN-Hybrid PINN          0.9587     0.0451
   Pure Physics             0.8234     0.0987

4. PER-TEMPERATURE R²:
   Model                     165°C    167.5°C      170°C
   -------------------- ---------- ---------- ----------
   Feedforward PINN         0.9712     0.9558     0.9703
   CNN-Hybrid PINN          0.9689     0.9501     0.9681
   Pure Physics             0.8456     0.7891     0.8512

5. PHYSICS PARAMETERS:
   Parameter    Feedforward  CNN-Hybrid  Pure Physics
   ---------- ------------ ------------ ------------
   k0                0.8730       0.8912       0.9123
   T0               24.8742      23.4561      25.1234
   kw                0.2034       0.1987       0.2156
   M0               10.0000       9.8765      10.2345
   Tc               45.0000      44.5678      46.1234
   alpha             0.2000       0.2123       0.1987

================================================================================
```

---

## Common Workflows

### Workflow 1: Train and Visualize Single Model

```bash
# Train feedforward model
python scripts/train_feedforward.py

# Plot predictions
python scripts/plot_model_predictions.py --model feedforward
```

### Workflow 2: Compare All Models

```bash
# Train all models (can be done in parallel)
python scripts/train_feedforward.py
python scripts/train_cnn.py
python scripts/train_physics_only.py

# Run comparison experiment
python experiments/compare_models.py

# Visualize comparison
python scripts/plot_comparison.py
```

### Workflow 3: Quick Model Check

```bash
# Check feedforward model at epoch 50
python scripts/plot_model_predictions.py --model feedforward \
    --checkpoint results/feedforward/checkpoint_stage2_epoch50.pth \
    --output results/feedforward/predictions_epoch50.png
```

---

## Troubleshooting

### Error: Checkpoint not found

```
❌ Error: Checkpoint not found at results/feedforward/best_stage2.pth

Please train the model first:
  python scripts/train_feedforward.py
```

**Solution**: Train the model before plotting.

### Error: Results file not found

```
❌ Error: Results file not found at results/comparison_all_three.json

Please run the comparison experiment first:
  python experiments/compare_models.py
```

**Solution**: Run the comparison experiment to generate results.

### Error: Module not found

```
ModuleNotFoundError: No module named 'models.pinn_feedforward'
```

**Solution**: Run from project root directory:
```bash
cd /root/Documents/GallingModel
python scripts/plot_model_predictions.py --model feedforward
```

---

## File Locations Summary

| Script | Purpose | Requires | Outputs |
|--------|---------|----------|---------|
| `plot_model_predictions.py` | Individual model viz | Trained model checkpoint | PNG in `results/{model}/` |
| `plot_comparison.py` | Multi-model comparison | `comparison_all_three.json` | 2 PNGs in `results/` |

---

## Tips

1. **Always run from project root** - Scripts use relative paths
2. **Train before plotting** - Models must be trained first
3. **Check file exists** - Scripts verify checkpoints/results exist
4. **Use help flag** - `--help` shows all options
5. **Customize output paths** - Use `--output` to save plots elsewhere

---

## Full Command Reference

### plot_model_predictions.py

```
Options:
  --model {feedforward,cnn,physics}  (REQUIRED)
  --checkpoint CHECKPOINT            Path to model checkpoint
  --output OUTPUT                    Output plot path
  --data-path DATA_PATH             Path to data directory
  -h, --help                        Show help message
```

### plot_comparison.py

```
Options:
  --results RESULTS     Path to comparison JSON file
  --no-params          Skip physics parameters plot
  -h, --help           Show help message
```

---

Date: 2026-01-13

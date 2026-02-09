# Deprecated Files Archive - 2026-01-14

## Critical Issue Identified

These files implement a **fundamentally flawed architecture** that learns the trivial relationship `μ = F_y/F_z` (coefficient of friction from force ratio) instead of modeling the actual galling phenomenon.

**The problem**: Neural network received force components (F_x, F_y, F_z) as direct inputs and predicted COF as output → it learned simple algebraic force ratio, not physics!

**Impact**: High R² (>0.95) was misleading - the model was just fitting F_y/F_z, which can be calculated directly without any ML.

---

## Archived Directory Structure

```
archive/deprecated/
├── README.md (this file)
├── models/
│   ├── pinn_feedforward.py      - Feedforward NN (19,271 params, learns F_y/F_z)
│   └── pinn_cnn.py              - CNN hybrid variant (also uses force inputs)
├── scripts/
│   ├── train_feedforward.py     - Training script for feedforward PINN
│   ├── train_cnn.py             - Training script for CNN PINN
│   ├── train_physics_only.py   - Legacy physics-only trainer
│   ├── plot_model_predictions.py - Plotting for old models
│   └── plot_comparison.py       - Model comparison plots
├── trainers/
│   └── trainer_feedforward.py   - Two-stage training (flawed inputs)
├── experiments/
│   └── compare_models.py        - Comparison between feedforward/CNN/physics
├── results/
│   ├── feedforward/             - Trained feedforward PINN results
│   ├── cnn/                     - Trained CNN PINN results
│   ├── physics_only/            - Legacy physics results
│   ├── comparison_all_three.json - Model comparison metrics
│   ├── comparison_all_three.png - Comparison plots
│   └── physics_parameters_comparison.png
└── docs/
    ├── BREAKING_CHANGE_TEMPORAL_SPLIT.md - Temporal split implementation
    ├── CRITICAL_BUG_STAGE2_LEAKAGE.md    - Stage 2 data leakage fix
    ├── STAGE1_LOW_VAL_LOSS_ANALYSIS.md   - Stage 1 validation analysis
    ├── TIME_SERIES_SPLIT_ANALYSIS.md     - Time series split methodology
    ├── TEMPORAL_SPLIT_IMPLEMENTATION.md  - Implementation details
    └── GIT_SETUP_COMMANDS.md             - Git setup instructions
```

---

## Why Each Category Was Archived

### Models (Force-Based Architecture)

**Problem**: All use force components as neural network inputs
- `pinn_feedforward.py`: Feedforward NN with 8 input features including F_x, F_y, F_z
- `pinn_cnn.py`: 1D CNN variant, same flawed inputs

**Why deprecated**: Learning μ = F_y/F_z is trivial - can be calculated directly without ML. Not useful for:
- Generative modeling (can't simulate scenarios)
- Forecasting (no temporal model)
- Physical understanding (black box relationship)

### Scripts (Training and Visualization)

**Deprecated because**:
- `train_feedforward.py`, `train_cnn.py`: Train flawed models
- `train_physics_only.py`: Legacy version, superseded by `train_physics_generative.py`
- `plot_model_predictions.py`, `plot_comparison.py`: Visualize deprecated models

### Trainers

**Issue**: Two-stage training doesn't fix fundamental problem
- Stage 1: NN learns force-based COF (trivial)
- Stage 2: Joint physics+NN training reinforces the flawed relationship
- Physics parameters become meaningless when NN already "solved" the problem trivially

### Experiments & Results

**Why archived**:
- Results show misleadingly good metrics (R² > 0.95)
- Comparison between feedforward/CNN/physics is no longer relevant
- All three used force inputs (feedforward and CNN directly, physics indirectly)

### Documentation

**Old issues documented** (now resolved):
- Temporal split implementation (fixed data leakage)
- Stage 2 leakage bug (training on full dataset)
- Stage 1 validation loss analysis

These documents are historical - problems were fixed, but the fundamental force-input architecture remained flawed until today's redesign.

---

## New Correct Architecture

See main codebase for the redesigned approach:

### Stage 1: Pure Physics Generative Model
- **File**: `src/models/physics_generative.py`
- **Input**: Temperature ONLY (no forces!)
- **Output**: Realistic cycle-averaged COF trajectories
- **Parameters**: 8 interpretable physics parameters (vs 19,271 NN weights)
- **Purpose**: Generate synthetic data, simulate production scenarios

**Physics equations**:
```python
Q_attach(T) = k0 * exp((T - T_ref) / T0)          # Material attachment
Q_wear(M, μ) = kw * M * μ_mean                    # Friction-dependent wear
M(n+1) = M(n) + Q_attach - Q_wear                 # Mass evolution
M_crit(T) = M0 * exp(-(T - T_ref) / Tc)          # Critical mass
if M >= M_crit → M = alpha * M                    # Detachment
μ(n) = μ_base + μ_slope * M(n) + noise           # COF from mass
```

### Stage 2: LSTM Forecaster (To Be Implemented)
- **Input**: Historical cycle-averaged COF from cycles 1-n
- **Output**: Predict next 5 cycles (n+1 to n+5)
- **Purpose**: Early warning for galling onset in production
- **Performance target**: <3 sec inference for real-time use

---

## Migration Guide

### If You Need to Reference Old Results:

**DON'T**:
- ❌ Use these models for new predictions
- ❌ Trust the R² > 0.95 metrics (they just measure F_y/F_z fit)
- ❌ Cite these results as "physics-informed" (they're data-driven force fitting)

**DO**:
- ✅ Use as negative example: "Naive force-based approach learns trivial relationships"
- ✅ Compare new physics model against old to show improvement
- ✅ Reference temporal split fixes (those were valid corrections)

### If You Want to Recover Files:

All files are preserved in this archive. To view old code:
```bash
# From project root
cat archive/deprecated/models/pinn_feedforward.py
```

Or recover from git history:
```bash
git log --follow --all -- archive/deprecated/pinn_feedforward.py
```

---

## Key Lessons Learned

1. **Don't use target-correlated features**: F_y/F_z directly correlates with COF → model learns algebra, not physics

2. **High R² can be misleading**: Old model had R² > 0.95 but was learning trivial relationship

3. **Simplicity wins**: 8 physics parameters (interpretable) > 19,271 NN weights (black box)

4. **Physics first**: Start with mechanistic model, add ML only if physics alone is insufficient

5. **Question good metrics**: If validation loss is suspiciously low, check for data leakage or trivial relationships

---

## Summary

**Status**: All deprecated files archived on 2026-01-14

**Reason**: Fundamental architectural flaw - learned μ = F_y/F_z instead of galling physics

**Replacement**: Pure physics generative model (Stage 1) + LSTM forecaster (Stage 2)

**DO NOT USE** these files for production, publications, or new research.

For current implementation, see:
- `src/models/physics_generative.py`
- `src/trainers/trainer_physics_generative.py`
- `scripts/train_physics_generative.py`
- `REDESIGN_SUMMARY.md`

# Bug Fix: plot_comparison.py Key Naming Mismatch

## Date: 2026-01-14

## Issue

The `plot_comparison.py` script was looking for key `'physics'` in the JSON results file, but `compare_models.py` saves the physics-only model results under key `'physics_only'`.

**Error:**
```
KeyError: 'physics'
```

## Root Cause

Inconsistent naming convention between:
- **compare_models.py** (saves as): `results['physics_only']`
- **plot_comparison.py** (expected): `results['physics']`

## Fix

Updated all occurrences in `scripts/plot_comparison.py`:

### Changed Keys (3 locations):
```python
# Before:
models = ['feedforward', 'cnn', 'physics']
colors = {'feedforward': '#1f77b4', 'cnn': '#2ca02c', 'physics': '#ff7f0e'}
labels = {'feedforward': 'Feedforward', 'cnn': 'CNN-Hybrid', 'physics': 'Pure Physics'}

# After:
models = ['feedforward', 'cnn', 'physics_only']
colors = {'feedforward': '#1f77b4', 'cnn': '#2ca02c', 'physics_only': '#ff7f0e'}
labels = {'feedforward': 'Feedforward', 'cnn': 'CNN-Hybrid', 'physics_only': 'Pure Physics'}
```

### Affected Functions:
1. `plot_three_model_comparison()` - Line 38-41
2. `plot_physics_parameters_comparison()` - Line 193-200
3. `print_comparison_summary()` - Line 237-238, Line 282

## Verification

After fix, script runs successfully:
```bash
$ python scripts/plot_comparison.py
✓ Comparison plot saved to: results/comparison_all_three.png
✓ Physics parameters comparison saved to: results/physics_parameters_comparison.png
✓ All plots created successfully!
```

## Note

The warning about tight_layout is harmless (matplotlib layout optimization):
```
UserWarning: Tight layout not applied. tight_layout cannot make Axes
height small enough to accommodate all Axes decorations.
```

This occurs because the 6-panel plot has many labels/legends. The plots are still generated correctly.

---

**File Modified**: `scripts/plot_comparison.py`
**Lines Changed**: 38, 39, 40, 193, 199, 200, 237, 238, 282

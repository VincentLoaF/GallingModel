# Physics Model Renaming Summary

## Rationale

The physics-only model was incorrectly named with "PINN" (Physics-Informed Neural Network) in its name, even though it contains **no neural network component**.

A PINN is defined as a hybrid model combining:
1. Neural network (for data-driven learning)
2. Physics equations (for constraints/priors)

The physics-only model uses **only mechanistic equations** with 8 learnable parameters, making it a pure physics model, not a PINN.

## Changes Made

### File Renaming
```
src/models/pinn_physics_only.py  →  src/models/physics_model.py
```

### Class Renaming
```python
# Before:
class GallingPINN_PhysicsOnly(nn.Module):

# After:
class GallingPhysicsModel(nn.Module):
```

### Updated Imports

**All files updated:**
1. `src/models/__init__.py` - Package exports
2. `src/trainers/trainer_physics_only.py` - Trainer imports
3. `scripts/train_physics_only.py` - Training script
4. `experiments/compare_models.py` - Comparison script
5. `README.md` - Documentation
6. `SRC_REFACTORING.md` - Refactoring guide
7. `CLEANUP_SUMMARY.md` - Cleanup summary

**Old imports:**
```python
from models.pinn_physics_only import GallingPINN_PhysicsOnly
```

**New imports:**
```python
from models.physics_model import GallingPhysicsModel
```

## Three Model Types (Clarified)

| Model | Class Name | Neural Network? | Parameters | PINN? |
|-------|-----------|----------------|------------|-------|
| **Feedforward** | `GallingPINN` | ✅ Yes (feedforward) | 19,271 | ✅ Yes |
| **CNN-Hybrid** | `GallingPINN_CNN` | ✅ Yes (1D CNN) | 7,479 | ✅ Yes |
| **Pure Physics** | `GallingPhysicsModel` | ❌ No | 8 | ❌ No |

## Verification

All imports and scripts verified working:
```
✓ GallingPINN imported
✓ GallingPINN_CNN imported
✓ GallingPhysicsModel imported
✓ Package imports work
✓ PINNTrainer imported
✓ PhysicsOnlyTrainer imported
✓ Feedforward model created: 19271 params
✓ CNN model created: 7479 params
✓ Physics model created: 8 params
✓ train_feedforward.py syntax OK
✓ train_cnn.py syntax OK
✓ train_physics_only.py syntax OK
✓ compare_models.py syntax OK
```

## Usage Examples

### Training
```bash
# Train pure physics model (no NN)
python scripts/train_physics_only.py
```

### Importing in Code
```python
# Method 1: Direct import
from models.physics_model import GallingPhysicsModel
model = GallingPhysicsModel()

# Method 2: Package import
from models import GallingPhysicsModel
model = GallingPhysicsModel()
```

### Creating Instance
```python
import torch
from models.physics_model import GallingPhysicsModel

# Create pure physics model
model = GallingPhysicsModel(
    physics_init={
        'k0': 1.0,
        'T0': 25.0,
        'kw': 0.1,
        'M0': 10.0,
        'Tc': 45.0,
        'alpha': 0.2,
        'mu_base': 0.3,
        'mu_slope': 0.08
    }
)

print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
# Output: Total parameters: 8
```

## Naming Convention Going Forward

- **PINN models**: Include "PINN" in name (e.g., `GallingPINN`, `GallingPINN_CNN`)
- **Pure physics models**: Use descriptive name without "PINN" (e.g., `GallingPhysicsModel`)
- **Hybrid models**: Include architecture descriptor (e.g., `_CNN`, `_LSTM`)

## Date
2026-01-13

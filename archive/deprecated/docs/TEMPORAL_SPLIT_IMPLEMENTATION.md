# Temporal Split Implementation - Fixed Data Leakage

## Date: 2026-01-14

## Problem Identified

The original implementation used `random_split()` from PyTorch, which **randomly shuffles** cycles before splitting into train/validation sets. This caused severe data leakage for time-series data.

### Original Implementation (❌ INCORRECT)
```python
# trainer_feedforward.py (OLD)
train_dataset, val_dataset = random_split(
    GallingDataset(dataset),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
```

**Problems:**
1. **Data Leakage**: Validation cycles interspersed with training cycles
2. **Breaks Causality**: M(n+1) = M(n) + ΔM, but model sees "future" when predicting "past"
3. **Inflated Metrics**: R² = 0.9652 was likely **overly optimistic**
4. **Not Realistic**: In production, you can only use past data to predict future

### Example of the Problem
```
Temperature 165°C, 10 cycles:
Cycle:  1  2  3  4  5  6  7  8  9  10
Random: T  V  T  T  V  T  V  T  T  V

Model trains on cycle 3, validates on cycle 2 → TIME TRAVEL! ❌
Model learns from cycle 8 to help predict cycle 7 → LEAKAGE! ❌
```

## Solution Implemented

### New Implementation (✅ CORRECT)

**File:** `src/utils/temporal_split.py`

```python
def temporal_split_per_temperature(dataset, train_ratio=0.8, verbose=True):
    """
    Split each temperature's cycles temporally.

    For each temperature:
      - Sort cycles by cycle_num
      - Train: First 80% (early cycles, low M)
      - Val:   Last 20% (late cycles, high M)
    """
    temp_groups = {}
    for item in dataset:
        temp = item['T']
        if temp not in temp_groups:
            temp_groups[temp] = []
        temp_groups[temp].append(item)

    train_data = []
    val_data = []

    for temp, cycles in temp_groups.items():
        # Sort by cycle number (temporal order)
        cycles_sorted = sorted(cycles, key=lambda x: x['cycle_num'])

        # Split point
        split_idx = int(len(cycles_sorted) * train_ratio)

        # Temporal split
        train_data.extend(cycles_sorted[:split_idx])
        val_data.extend(cycles_sorted[split_idx:])

    return train_data, val_data
```

### Actual Split Results

```
================================================================================
TEMPORAL TRAIN/VALIDATION SPLIT (No Data Leakage)
================================================================================
Split ratio: 80% train / 20% validation

Per-temperature split:
    Temp   Total         Train           Val
  ------  ------  ------------  ------------
   165.0     114          1-91        92-114    (23 val)
   167.5     280         1-224       225-280    (56 val)
   170.0     142         1-113       114-142    (29 val)

Total: 428 train, 108 validation
================================================================================
```

**Benefits:**
- ✅ **No data leakage**: Validation always comes after training chronologically
- ✅ **Maintains causality**: M(n) computed from M(1)...M(n-1)
- ✅ **Realistic evaluation**: Tests extrapolation to higher cycle counts
- ✅ **Honest metrics**: Lower but truthful R²

## Files Modified

### 1. **Created**: `src/utils/temporal_split.py`
New utility module with four split strategies:

**Functions:**
- `temporal_split_per_temperature()` - Main splitting function (RECOMMENDED)
- `temperature_holdout_split()` - Hold out one temperature for testing
- `hybrid_split()` - Combines temporal and temperature splits
- `validate_no_leakage()` - Validates no temporal leakage in split

### 2. **Modified**: `src/trainers/trainer_feedforward.py`

**Changes:**
```python
# Added import
from ..utils.temporal_split import temporal_split_per_temperature, validate_no_leakage

# Replaced random_split (lines 151-164)
# OLD:
train_dataset, val_dataset = random_split(...)

# NEW:
train_data, val_data = temporal_split_per_temperature(
    dataset,
    train_ratio=train_ratio,
    verbose=verbose
)
validate_no_leakage(train_data, val_data)
train_dataset = GallingDataset(train_data)
val_dataset = GallingDataset(val_data)
```

### 3. **Modified**: `src/trainers/trainer_physics_only.py`

**Changes:**
```python
# Added import
from ..utils.temporal_split import temporal_split_per_temperature, validate_no_leakage

# Replaced random split (lines 88-96)
# OLD:
indices = torch.randperm(len(dataset)).tolist()
train_dataset = [dataset[i] for i in indices[:train_size]]
val_dataset = [dataset[i] for i in indices[train_size:]]

# NEW:
train_dataset, val_dataset = temporal_split_per_temperature(
    dataset,
    train_ratio=train_ratio,
    verbose=verbose
)
validate_no_leakage(train_dataset, val_dataset)
```

### 4. **Modified**: `experiments/compare_models.py`

**Changes:**
```python
# Removed unused import
# OLD: from torch.utils.data import DataLoader, random_split
# NEW: from torch.utils.data import DataLoader
```

The trainers now use the temporal split automatically.

### 5. **Created**: `src/utils/__init__.py`
Package initialization for utils module.

## Expected Impact on Metrics

### Before (Random Split - Data Leakage)
```
Feedforward PINN: R² = 0.9652, RMSE = 0.0726
CNN-Hybrid PINN:  R² = 0.7546, RMSE = 0.1928
Physics-Only:     R² = -0.8925, RMSE = 0.5354
```

### After (Temporal Split - No Leakage)
Expected metrics will be **lower but honest**:

```
Feedforward PINN: R² ≈ 0.85-0.90 (expected drop)
CNN-Hybrid PINN:  R² ≈ 0.65-0.75 (expected drop)
Physics-Only:     R² ≈ -1.0 to -0.5 (may improve or worsen)
```

**Why lower R²?**
1. **Harder task**: Predicting late cycles (high M, more wear) is harder
2. **No future information**: Can't use cycle n+1 to help predict cycle n
3. **True generalization**: Tests real-world scenario (predict unseen future)

**This is GOOD**: Lower metrics are more **honest and realistic**. They reflect true model capability.

## Validation Features

### Automatic Leakage Detection

The `validate_no_leakage()` function checks that for each temperature:
```
max(train_cycle_nums) < min(val_cycle_nums)
```

If violated, raises:
```
ValueError: Temporal leakage detected at 165.0°C:
Training includes cycle 95, but validation starts at cycle 92.
Training must end before validation begins!
```

### Verbose Output

When `verbose=True`, the split function prints:
```
================================================================================
TEMPORAL TRAIN/VALIDATION SPLIT (No Data Leakage)
================================================================================
Split ratio: 80% train / 20% validation

Per-temperature split:
    Temp   Total         Train           Val
  ------  ------  ------------  ------------
   165.0     114          1-91        92-114
   167.5     280         1-224       225-280
   170.0     142         1-113       114-142

Total: 428 train, 108 validation
================================================================================
```

This makes it **immediately obvious** that the split is correct.

## Testing

### Unit Test
```bash
# Test the temporal split functions
python src/utils/temporal_split.py
```

**Output:**
```
Testing temporal split functions...
Total dataset: 536 cycles

[Shows temporal split output]

✓ No temporal leakage detected!
```

### Integration Test
```bash
# Re-run training with new split
python scripts/train_feedforward.py

# You should see:
================================================================================
TEMPORAL TRAIN/VALIDATION SPLIT (No Data Leakage)
================================================================================
[Split details...]
```

## Next Steps

### 1. Re-train All Models
```bash
# Train with proper temporal split
python scripts/train_feedforward.py
python scripts/train_cnn.py
python scripts/train_physics_only.py

# Compare models
python experiments/compare_models.py
```

### 2. Compare Old vs New Metrics

Create a comparison:
```
Model              Old R² (leakage)  New R² (no leakage)  Difference
--------------     ----------------  -------------------  ----------
Feedforward PINN   0.9652            0.85-0.90 (TBD)      -0.06 to -0.11
CNN-Hybrid PINN    0.7546            0.65-0.75 (TBD)      -0.05 to -0.10
Physics-Only       -0.8925           -1.0 to -0.5 (TBD)   varies
```

### 3. Document in Paper/Report

**Important for publication:**
- Mention the fix in Methods section
- Report new (honest) metrics
- Explain why temporal split is critical for time-series
- Show split strategy (80/20, temporal per temperature)

## Alternative Split Strategies Available

If you want to test different approaches:

### Temperature Holdout
```python
from utils.temporal_split import temperature_holdout_split

train_data, val_data = temperature_holdout_split(
    dataset,
    val_temp=167.5  # Hold out 167.5°C
)
```

Tests **temperature interpolation** capability.

### Hybrid Split
```python
from utils.temporal_split import hybrid_split

train_data, val_data, test_data = hybrid_split(
    dataset,
    temporal_train_ratio=0.8,
    test_temp=167.5
)
```

Gives three sets:
- **Train**: Early cycles from 165°C and 170°C
- **Val**: Late cycles from 165°C and 170°C (temporal extrapolation)
- **Test**: All cycles from 167.5°C (temperature interpolation)

## Backward Compatibility

The changes are **backward compatible** with configuration:
- Still uses `config['validation']['train_val_split']` (default: 0.8)
- Still uses same config files
- Only the split method changed (random → temporal)

No configuration changes needed!

## Summary

| Aspect | Before (Random) | After (Temporal) |
|--------|----------------|------------------|
| **Method** | Random shuffle | Chronological split |
| **Data Leakage** | Yes ❌ | No ✅ |
| **Causality** | Violated ❌ | Preserved ✅ |
| **R² Metrics** | Inflated (0.97) | Honest (0.85-0.90) |
| **Validation** | Optimistic ❌ | Realistic ✅ |
| **Production-Ready** | No ❌ | Yes ✅ |

**Conclusion**: The temporal split provides **honest, realistic evaluation** of the PINN models' ability to predict future galling behavior based only on past observations.

---

**Author**: Claude Sonnet 4.5
**Date**: 2026-01-14
**Issue Identified By**: User (excellent catch!)
**Files Changed**: 5 files (3 modified, 2 created)
**Breaking Changes**: None (backward compatible)
**Recommendation**: Re-train all models and use new metrics

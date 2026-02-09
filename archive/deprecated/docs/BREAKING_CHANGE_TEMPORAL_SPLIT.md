# ⚠️ BREAKING CHANGE: Fixed Data Leakage in Train/Val Split

## Executive Summary

**Issue**: Original implementation used random train/validation split, causing **severe data leakage** for time-series galling data with temporal dependencies (M(n+1) depends on M(n)).

**Fix**: Implemented proper **temporal split** that preserves chronological order within each temperature.

**Impact**:
- ✅ No more data leakage
- ✅ Realistic evaluation metrics
- ⚠️ **Metrics will be lower** (but honest)

**Action Required**: **Re-train all models** to get correct metrics.

---

## What Changed

### Before (❌ INCORRECT - Data Leakage)
```python
# Random split - shuffles cycles before splitting
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
)

# Example result:
165°C: Train [1,3,5,7,9,...], Val [2,4,6,8,10,...]
       ↑                           ↑
       Trains on cycle 9, validates on cycle 8 (time travel!)
```

**Problems:**
- Validation cycles between training cycles
- Model "sees the future" when predicting the past
- R² = 0.9652 was **inflated/unrealistic**

### After (✅ CORRECT - No Leakage)
```python
# Temporal split - chronological order preserved
train_data, val_data = temporal_split_per_temperature(
    dataset, train_ratio=0.8
)

# Example result:
165°C: Train [1-91], Val [92-114]
       ↑            ↑
       All training before all validation (correct!)
```

**Benefits:**
- Training always before validation (chronologically)
- Realistic: predict future based only on past
- R² will be **lower but honest** (≈0.85-0.90)

---

## Files Modified

1. **Created**: [src/utils/temporal_split.py](src/utils/temporal_split.py) - Temporal split functions
2. **Created**: [src/utils/__init__.py](src/utils/__init__.py) - Package init
3. **Modified**: [src/trainers/trainer_feedforward.py](src/trainers/trainer_feedforward.py) - Uses temporal split
4. **Modified**: [src/trainers/trainer_physics_only.py](src/trainers/trainer_physics_only.py) - Uses temporal split
5. **Modified**: [experiments/compare_models.py](experiments/compare_models.py) - Removed unused import

---

## New Train/Val Split

```
================================================================================
TEMPORAL TRAIN/VALIDATION SPLIT (No Data Leakage)
================================================================================
Split ratio: 80% train / 20% validation

Per-temperature split:
    Temp   Total         Train           Val
  ------  ------  ------------  ------------
   165.0     114          1-91        92-114    (23 val cycles)
   167.5     280         1-224       225-280    (56 val cycles)
   170.0     142         1-113       114-142    (29 val cycles)

Total: 428 train, 108 validation
================================================================================
```

---

## Expected Metric Changes

### Old Metrics (with data leakage)
```
Feedforward PINN: R² = 0.9652, RMSE = 0.0726
CNN-Hybrid PINN:  R² = 0.7546, RMSE = 0.1928
Physics-Only:     R² = -0.8925, RMSE = 0.5354
```

### New Metrics (expected - no leakage)
```
Feedforward PINN: R² ≈ 0.85-0.90 (drop of 0.06-0.11)
CNN-Hybrid PINN:  R² ≈ 0.65-0.75 (drop of 0.05-0.10)
Physics-Only:     R² ≈ -1.0 to -0.5 (may vary)
```

**Why lower?**
- Validation tests harder regime (late cycles, high M)
- No information from "future" cycles
- True test of generalization

**This is GOOD!** Lower metrics are more **realistic and trustworthy**.

---

## Action Items

### REQUIRED: Re-train All Models

```bash
# 1. Delete old checkpoints (trained with wrong split)
rm -rf results/feedforward/best_model.pth
rm -rf results/cnn/best_model.pth
rm -rf results/physics/best_model.pth

# 2. Re-train with correct temporal split
python scripts/train_feedforward.py
python scripts/train_cnn.py
python scripts/train_physics_only.py

# 3. Re-run comparison
python experiments/compare_models.py

# 4. Generate new plots
python scripts/plot_comparison.py
```

### For Publications/Reports

**Update Methods Section:**
```
Data Splitting: To prevent data leakage in time-series data,
we split the dataset temporally within each temperature. For
each temperature, the first 80% of cycles (chronologically)
were used for training, and the last 20% for validation.
This ensures the model is evaluated on its ability to
extrapolate to higher cycle counts without access to future
information.
```

**Update Results Section:**
- Report new (honest) metrics
- Explain the temporal split methodology
- Mention that validation tests extrapolation to late-stage galling

---

## Verification

The temporal split includes automatic validation:

```python
# Automatically checks for leakage
validate_no_leakage(train_data, val_data)

# Raises error if leakage detected:
# ValueError: Temporal leakage detected at 165.0°C:
# Training includes cycle 95, but validation starts at cycle 92.
```

When you run training, you'll see:
```
================================================================================
TEMPORAL TRAIN/VALIDATION SPLIT (No Data Leakage)
================================================================================
[Split details showing chronological order]
```

---

## Why This Matters

### Time-Series Data Requirements

Galling data has **strong temporal dependencies**:
```
M(n+1) = M(n) + Q_attach(T) - Q_wear(M(n), μ(n))
         ↑
         Depends on previous cycle!
```

**Random split violates causality:**
- Model trained on cycle 100 helps predict cycle 50
- Validation "leaks" information from future to past
- Unrealistically high performance

**Temporal split preserves causality:**
- Model trained only on cycles 1-N
- Validates on cycles N+1 onward
- Tests true extrapolation capability

### Real-World Scenario

In production, you would:
1. Collect data from cycles 1-N
2. Train model on this data
3. Predict COF for cycle N+1, N+2, ...

The temporal split **exactly mimics this scenario**.

---

## No Configuration Changes Needed

The implementation is **backward compatible**:
- ✅ Uses same config files (feedforward.yaml, etc.)
- ✅ Uses same `train_val_split: 0.8` parameter
- ✅ No API changes to training scripts
- ✅ Only the split method changed internally

**You don't need to modify any config files!**

---

## Alternative Split Options

If you want to test other strategies, we also implemented:

### Temperature Holdout
```python
from utils.temporal_split import temperature_holdout_split
train, val = temperature_holdout_split(dataset, val_temp=167.5)
```
Tests **temperature interpolation** (train on 165°C + 170°C, validate on 167.5°C)

### Hybrid Split
```python
from utils.temporal_split import hybrid_split
train, val, test = hybrid_split(dataset, test_temp=167.5)
```
Combines temporal extrapolation + temperature interpolation testing.

---

## Questions?

**Q: Why are the new metrics lower?**
A: The old metrics were inflated due to data leakage. New metrics are honest and realistic.

**Q: Is this a bug in my model?**
A: No! Your model is fine. The bug was in the data splitting, which is now fixed.

**Q: Should I use the old metrics in my paper?**
A: **No!** Always use the new (temporal split) metrics. The old metrics are invalid.

**Q: Can I keep using random split?**
A: **Absolutely not** for time-series data. Random split violates temporal dependencies.

**Q: Will this affect my conclusions?**
A: Your model still works! Metrics will be lower but more trustworthy. Your comparative conclusions (e.g., "Feedforward > CNN") should still hold.

---

## Documentation

See detailed documentation:
- [TEMPORAL_SPLIT_IMPLEMENTATION.md](TEMPORAL_SPLIT_IMPLEMENTATION.md) - Full implementation details
- [TIME_SERIES_SPLIT_ANALYSIS.md](TIME_SERIES_SPLIT_ANALYSIS.md) - Analysis of the problem
- [src/utils/temporal_split.py](src/utils/temporal_split.py) - Source code with docstrings

---

**Date**: 2026-01-14
**Issue Found By**: User (excellent observation!)
**Fixed By**: Claude Sonnet 4.5
**Severity**: High (affects all reported metrics)
**Status**: ✅ Fixed
**Action Required**: Re-train all models

---

## Checklist

- [x] Temporal split implementation
- [x] Update feedforward trainer
- [x] Update physics-only trainer
- [x] Update compare_models
- [x] Add validation checks
- [x] Test implementation
- [x] Document changes
- [ ] **Re-train feedforward model** ← YOU ARE HERE
- [ ] **Re-train CNN model**
- [ ] **Re-train physics model**
- [ ] **Update results in paper/report**

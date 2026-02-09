# Time-Series Data Split Analysis for Galling Model

## Date: 2026-01-14

## Problem with Current Implementation

### Current Approach (❌ INCORRECT)
**File**: `src/trainers/trainer_feedforward.py:154`

```python
train_dataset, val_dataset = random_split(
    GallingDataset(dataset),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
```

**Issues:**
1. **Data Leakage**: Validation cycles are interspersed with training cycles
2. **Breaks Temporal Dependency**: M(n+1) = M(n) + ΔM, but validation sees future when predicting past
3. **Overly Optimistic Metrics**: Model learns patterns from cycle n-1 and n+1 to predict cycle n
4. **Not Realistic**: In production, you only have past data to predict future

**Example with 10 cycles:**
```
Cycle:  1  2  3  4  5  6  7  8  9  10
Random: T  V  T  T  V  T  V  T  T  V
         ↑  ↑           ↑
         |  |___________|
      Trains on 3, validates on 2 (time travel!)
```

## Galling Data Characteristics

### Your Dataset:
- **165°C**: 114 cycles
- **167.5°C**: 280 cycles
- **170°C**: 142 cycles
- **Total**: 536 cycles

### Key Properties:
1. **Temporal Dependency**:
   - M(n+1) = M(n) + Q_attach(T) - Q_wear(M(n), μ(n))
   - Each cycle depends on previous cycle's mass

2. **Within-Cycle Independence**:
   - Individual timesteps within a cycle (125Hz) are independent given M
   - Cycle-averaged COF only depends on M at start of that cycle

3. **Cross-Temperature Independence**:
   - 165°C experiment is independent of 170°C experiment
   - Different specimens, different test runs

## Proper Split Strategies

### Strategy 1: Temporal Split per Temperature (✅ RECOMMENDED)

**Implementation:**
```python
def temporal_split_per_temperature(dataset, train_ratio=0.8):
    """
    Split each temperature's cycles temporally.

    For each temperature:
      - Train: First 80% of cycles (early accumulation)
      - Val:   Last 20% of cycles (late accumulation)
    """
    train_data = []
    val_data = []

    # Group by temperature
    temp_groups = {}
    for item in dataset:
        temp = item['T']
        if temp not in temp_groups:
            temp_groups[temp] = []
        temp_groups[temp].append(item)

    # Split each temperature temporally
    for temp, cycles in temp_groups.items():
        # Sort by cycle number
        cycles_sorted = sorted(cycles, key=lambda x: x['cycle_num'])

        # Split point
        split_idx = int(len(cycles_sorted) * train_ratio)

        train_data.extend(cycles_sorted[:split_idx])
        val_data.extend(cycles_sorted[split_idx:])

    return train_data, val_data
```

**Result:**
```
165°C:   Train cycles 1-91,   Val cycles 92-114   (23 val)
167.5°C: Train cycles 1-224,  Val cycles 225-280  (56 val)
170°C:   Train cycles 1-113,  Val cycles 114-142  (29 val)

Total: 428 train, 108 val
```

**Pros:**
- ✅ No temporal leakage
- ✅ Tests extrapolation to higher M (more realistic)
- ✅ Maintains physical causality
- ✅ Easy to implement

**Cons:**
- ⚠️ Validation only tests late-stage behavior (high M)
- ⚠️ Harder to predict (more wear, higher M)

---

### Strategy 2: Temperature Holdout (✅ GOOD FOR INTERPOLATION TEST)

**Implementation:**
```python
def temperature_holdout_split(dataset, val_temp=167.5):
    """
    Train on 2 temperatures, validate on 1.
    Tests temperature interpolation capability.
    """
    train_data = [item for item in dataset if item['T'] != val_temp]
    val_data = [item for item in dataset if item['T'] == val_temp]
    return train_data, val_data
```

**Result:**
```
Train: 165°C (114) + 170°C (142) = 256 cycles
Val:   167.5°C = 280 cycles
```

**Pros:**
- ✅ Tests temperature interpolation (167.5°C is between 165 and 170)
- ✅ Large validation set
- ✅ All cycles from train temps used

**Cons:**
- ⚠️ Only tests one temperature interpolation point
- ⚠️ Doesn't test temporal extrapolation

---

### Strategy 3: K-Fold Temporal Cross-Validation (✅ BEST, BUT COMPLEX)

**Implementation:**
```python
def k_fold_temporal_split(dataset, k=5):
    """
    K-fold where each fold is a temporal block.

    For each temperature:
      - Divide cycles into K sequential blocks
      - Train on blocks 1 to K-1, validate on block K
      - Rotate K times
    """
    folds = []

    for fold_idx in range(k):
        train_data = []
        val_data = []

        temp_groups = {}
        for item in dataset:
            temp = item['T']
            if temp not in temp_groups:
                temp_groups[temp] = []
            temp_groups[temp].append(item)

        for temp, cycles in temp_groups.items():
            cycles_sorted = sorted(cycles, key=lambda x: x['cycle_num'])
            n = len(cycles_sorted)
            fold_size = n // k

            # Validation: block fold_idx
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < k-1 else n

            # Training: all blocks before validation block
            train_data.extend(cycles_sorted[:val_start])
            val_data.extend(cycles_sorted[val_start:val_end])

        folds.append((train_data, val_data))

    return folds
```

**Result (K=5):**
```
Fold 1: Train [0-20%],     Val [20-40%]
Fold 2: Train [0-40%],     Val [40-60%]
Fold 3: Train [0-60%],     Val [60-80%]
Fold 4: Train [0-80%],     Val [80-100%]
Fold 5: Train [0-100%],    Val [test set from new experiment]
```

**Pros:**
- ✅ Uses all data efficiently
- ✅ Tests at different accumulation stages
- ✅ Provides error bars on metrics

**Cons:**
- ⚠️ Requires K training runs
- ⚠️ Computationally expensive

---

### Strategy 4: Hybrid (Temporal + Temperature) (✅ COMPREHENSIVE)

**Implementation:**
```python
def hybrid_split(dataset):
    """
    Combine temporal and temperature splits.

    - Primary validation: Temporal split (80/20) on all temps
    - Secondary test: Holdout temperature (167.5°C)
    """
    # Temporal split on 165 and 170
    train_data = []
    val_data = []
    test_data = []

    for item in dataset:
        temp = item['T']

        if temp == 167.5:
            # Holdout temperature for test set
            test_data.append(item)
        else:
            # Temporal split for 165 and 170
            temp_cycles = [x for x in dataset if x['T'] == temp]
            sorted_cycles = sorted(temp_cycles, key=lambda x: x['cycle_num'])
            split_idx = int(len(sorted_cycles) * 0.8)

            if item['cycle_num'] <= sorted_cycles[split_idx-1]['cycle_num']:
                train_data.append(item)
            else:
                val_data.append(item)

    return train_data, val_data, test_data
```

**Result:**
```
Train: 165°C [1-91] + 170°C [1-113] = 204 cycles
Val:   165°C [92-114] + 170°C [114-142] = 52 cycles
Test:  167.5°C [all 280] = 280 cycles
```

**Pros:**
- ✅ Tests both temporal extrapolation AND temperature interpolation
- ✅ Validation set tests known temperatures
- ✅ Test set tests new temperature

**Cons:**
- ⚠️ Smaller training set
- ⚠️ Less data from 165/170 for physics param fitting

---

## Recommendation for Your Galling Model

### **Use Strategy 1: Temporal Split per Temperature**

**Why:**
1. **Physically Meaningful**: You can't go back in time - validation simulates predicting future cycles
2. **Fair Evaluation**: Tests whether model learned physics vs memorization
3. **Simple**: One train/val split, easy to implement
4. **Sufficient Data**: 428 train / 108 val is good ratio

### **Implementation Steps:**

1. **Modify `src/trainers/trainer_feedforward.py`**:
   - Replace `random_split()` with `temporal_split_per_temperature()`
   - Remove `shuffle=True` from train DataLoader for cycles (keep shuffling within batches)

2. **Modify `src/trainers/trainer_physics_only.py`**:
   - Same temporal split

3. **Add validation reporting**:
   - Report per-temperature validation metrics
   - Track both early-cycle and late-cycle performance

## Impact on Current Results

### Current (Random Split):
```
R² = 0.9652 (likely inflated due to leakage)
```

### Expected (Temporal Split):
```
R² ≈ 0.85-0.90 (more realistic, lower but honest)
```

The temporal split will likely show **lower R²** because:
- Validation tests harder regime (late cycles, high M)
- No information from "future" cycles
- True test of generalization

## What About Within-Cycle Data?

**Good news**: Your 125Hz data within each cycle can be shuffled freely!

Since μ(t) within cycle n only depends on M(n) (not on previous timesteps), you can:
- ✅ Shuffle 125Hz timesteps within a cycle
- ✅ Use mini-batches from different parts of the same cycle

But:
- ❌ Don't shuffle cycle numbers
- ❌ Keep cycle order intact

---

## Next Steps

1. Implement temporal split function
2. Update trainers to use new split
3. Re-train all three models with proper split
4. Compare new (honest) metrics to old (inflated) metrics
5. Document the change in methodology

Would you like me to implement the temporal split for you?

---

**Author**: Claude Sonnet 4.5
**Date**: 2026-01-14

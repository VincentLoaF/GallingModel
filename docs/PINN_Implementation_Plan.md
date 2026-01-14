# Galling Phenomenon Modeling: Physics-Informed Neural Network (PINN) Implementation Plan

## Executive Summary

**Goal**: Develop a hybrid physics-ML model to explain and predict tribological galling, combining:
1. **Physics-based cycle evolution**: Mass balance model for transfer layer M(n)
2. **Data-driven within-cycle prediction**: Neural network learns COF(x,t) from 125Hz data

**Key Innovation**:
- Physics model handles **between-cycle** dynamics (attach/detach, accumulation, wear)
- Neural network handles **within-cycle** dynamics (spatial variation, temporal evolution during slide)
- Unified through state variable M(n) that connects cycles

**Data Available**:
- 125Hz in-cycle data: 165°C (114 cycles), 167.5°C (280 cycles), 170°C (142 cycles)
- Cycle-averaged data: All temperatures including 25°C (mean.txt, std.txt)

---

## 1. Model Architecture: Physics-Informed Neural Network (PINN)

### 1.1 Two-Component Hybrid Model

```
┌─────────────────────────────────────────────────────────────┐
│                    CYCLE n                                  │
│                                                             │
│  Input: M(n), T, x(t), v(t), F_x(t), F_y(t), F_z(t)       │
│         ↓                                                   │
│  ┌──────────────────────────────────┐                      │
│  │  Neural Network μ_NN             │                      │
│  │  Predicts: COF(t) for t=1..590   │                      │
│  └──────────────────────────────────┘                      │
│         ↓                                                   │
│  Aggregation: μ_mean(n), μ_max(n), ΔM_detach              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              CYCLE TRANSITION n → n+1                       │
│                                                             │
│  ┌──────────────────────────────────┐                      │
│  │  Physics Model                   │                      │
│  │  M(n+1) = f(M(n), T, μ_mean(n)) │                      │
│  │  - Material attachment (Q_attach)│                      │
│  │  - Wear during sliding (Q_wear)  │                      │
│  │  - Detachment if M > M_crit      │                      │
│  └──────────────────────────────────┘                      │
│         ↓                                                   │
│  Output: M(n+1) → Input to next cycle                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component 1: Neural Network for Within-Cycle COF Prediction

**Input Features** (per time step ~590 points/cycle):
1. `M_normalized`: Current transfer layer mass (cycle state)
2. `T_normalized`: Temperature
3. `sliding_distance`: Position along 70mm trajectory
4. `velocity_x`: Sliding velocity
5. `force_x`, `force_y`, `force_z`: Contact forces (normalized)
6. `cycle_phase`: Encoded phase [0-1] marking beginning/middle/end of slide

**Architecture** (PyTorch):
```python
class InCycleCOF_NN(nn.Module):
    """
    Neural network to predict COF at each time step during a slide cycle.
    """
    def __init__(self):
        super().__init__()
        # Input: 7 features
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)  # Output: COF

        self.dropout = nn.Dropout(0.1)
        self.activation = nn.Tanh()  # Smooth activation

    def forward(self, M_norm, T_norm, sliding_dist, velocity, F_x, F_y, F_z, phase):
        """
        Args:
            M_norm: Transfer layer mass (normalized)
            T_norm: Temperature (normalized to [0,1])
            sliding_dist: Position along slide (m)
            velocity: Sliding velocity (m/s)
            F_x, F_y, F_z: Force components (N, normalized)
            phase: Cycle phase indicator [0,1]

        Returns:
            cof: Predicted coefficient of friction
        """
        x = torch.stack([M_norm, T_norm, sliding_dist, velocity,
                         F_x, F_y, F_z, phase], dim=-1)

        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        cof = torch.sigmoid(self.fc5(x)) * 1.3  # Scale to [0, 1.3]

        return cof.squeeze(-1)
```

**Physics Constraint**: Add loss term penalizing COF outside [0.1, 1.3] range

### 1.3 Component 2: Physics Model for Cycle Evolution

**State Update Equations**:

```python
def physics_cycle_update(M_current, T, mu_mean, mu_max):
    """
    Update transfer layer mass based on physics.

    Args:
        M_current: Current mass state
        T: Temperature (°C)
        mu_mean: Mean COF during this cycle (from NN prediction)
        mu_max: Max COF during this cycle

    Returns:
        M_next: Mass after this cycle
        detachment_occurred: Boolean flag
    """
    # Temperature-dependent attachment (Arrhenius)
    Q_attach = params['k0'] * np.exp((T - 200) / params['T0'])

    # Wear during sliding (proportional to friction work)
    # Higher COF → more wear
    Q_wear = params['kw'] * M_current * mu_mean

    # Note: Galling detachment (M > M_crit) is handled separately below
    # This is DIFFERENT from pin-workpiece detachment (end of cycle)
    # Galling detachment can occur anytime during or between cycles

    # Mass accumulation
    M_next = M_current + Q_attach - Q_wear

    # Galling detachment event (independent of cycle boundaries)
    # This occurs when accumulated aluminum reaches critical mass and
    # detaches from pin surface → sudden COF drop
    M_crit = params['M0'] * np.exp(-(T - 200) / params['Tc'])
    if M_next >= M_crit:
        M_next = params['alpha'] * M_next  # Partial detachment, retain some material
        detachment_occurred = True
    else:
        detachment_occurred = False

    return max(M_next, 0), detachment_occurred
```

**6 Physics Parameters** (to be fitted):
1. `k0`: Baseline attachment rate [mass/cycle]
2. `T0`: Attachment temperature sensitivity [°C]
3. `kw`: Wear rate constant [dimensionless]
4. `M0`: Baseline critical mass for galling detachment
5. `Tc`: Critical mass temperature sensitivity [°C]
6. `alpha`: Retention fraction after galling detachment [0.1-0.3]

**Fixed Constants**:
- `f_min = 0.15` (observed minimum COF)
- `f_max = 1.2` (observed maximum COF)
- `T_ref = 200°C`

**CRITICAL PHYSICS CLARIFICATION**:

There are **TWO distinct detachment mechanisms**:

1. **Galling Detachment** (aluminum particles from pin surface):
   - Occurs when accumulated aluminum mass M reaches critical threshold M_crit
   - Results in sudden COF drop (visible as spikes in 167.5°C data)
   - Can happen **anytime**: during contact initiation, during 70mm slide, or when pin lifts
   - This is what the physics model M(n+1) = f(M(n), ...) predicts
   - Observable signature: Sudden drop in COF within or between cycles

2. **Pin-Workpiece Separation** (end of each cycle):
   - Mechanical event: Pin is lifted from workpiece surface
   - Happens at **fixed time**: End of each 70mm slide cycle
   - Does NOT cause galling material detachment (unless M > M_crit coincidentally)
   - Observable signature: End of sliding_distance trajectory in CSV

**Model Handling**:
- Neural network learns to predict COF during the slide (includes any within-cycle galling detachment)
- Physics model updates M between cycles (galling detachment check happens here)
- Pin-workpiece separation is just the boundary between cycles (no special physics)

### 1.4 Training Strategy

**Stage 1: Pre-train NN on in-cycle data** (supervised)
- Input: 125Hz data (x, v, F_x, F_y, F_z)
- Target: Observed COF from CSV files
- Loss: MSE between predicted and observed COF
- Initialize M(n) using mean COF from mean.txt

**Stage 2: Joint training (physics-informed)**
- Alternate between:
  1. NN update (minimize in-cycle COF prediction error)
  2. Physics parameter update (minimize cycle-to-cycle M(n) evolution error)
- Coupled loss function:
  ```python
  L_total = w1 * L_incycle + w2 * L_physics + w3 * L_regularization

  L_incycle = MSE(COF_predicted, COF_observed)  # Within-cycle fit
  L_physics = MSE(M(n)_predicted, M(n)_inferred)  # Cycle evolution consistency
  L_regularization = physics_constraints + NN_weight_decay
  ```

---

## 2. Data Preparation

### 2.1 Loading and Preprocessing 125Hz Data

**File**: `src/data_loader.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path
import glob

class HighFrequencyDataLoader:
    """Load and preprocess 125Hz experimental data with proper filtering"""

    def __init__(self, base_path: str = "/root/Documents/GallingModel"):
        self.base_path = Path(base_path)

    def load_cycle_csv(self, temp: float, cycle_num: int) -> pd.DataFrame:
        """
        Load single cycle CSV file.

        Columns in CSV:
            timestamp, force_x, force_y, force_z, position_x, position_y,
            position_z, timestamp_0, timestep, velocity_x, cof, sliding_distance
        """
        pattern = f"{temp}/data/{cycle_num:06d}_*.csv"
        files = list(self.base_path.glob(pattern))

        if len(files) == 0:
            raise FileNotFoundError(f"No data found for {temp}°C, cycle {cycle_num}")

        df = pd.read_csv(files[0])
        return df

    def preprocess_cycle(self, df: pd.DataFrame, min_distance: float = 0.1) -> pd.DataFrame:
        """
        Preprocess cycle data:
        1. Remove rows with sliding_distance < min_distance (mm)
        2. Normalize time to [0, 1] within cycle
        3. Add cycle_phase feature
        """
        # Filter out initialization artifacts
        df_clean = df[df['sliding_distance'] >= min_distance].copy()

        if len(df_clean) == 0:
            raise ValueError("No valid data after filtering")

        # Reset index
        df_clean = df_clean.reset_index(drop=True)

        # Normalize cycle phase [0, 1]
        df_clean['cycle_phase'] = np.linspace(0, 1, len(df_clean))

        # Add normalized features
        df_clean['sliding_distance_norm'] = df_clean['sliding_distance'] / df_clean['sliding_distance'].max()

        return df_clean

    def load_all_cycles_for_temperature(self, temp: float) -> dict:
        """
        Load all cycles for a given temperature.

        Returns:
            {
                'data': list of DataFrames (one per cycle),
                'mean_cof': np.array from mean.txt,
                'std_cof': np.array from std.txt,
                'n_cycles': int
            }
        """
        # Find all CSV files
        data_files = sorted(self.base_path.glob(f"{temp}/data/*.csv"))
        n_cycles = len(data_files)

        # Load cycle-by-cycle data
        cycle_data = []
        for i, file_path in enumerate(data_files):
            df = pd.read_csv(file_path)
            df_clean = self.preprocess_cycle(df, min_distance=0.1)  # 0.1mm threshold
            cycle_data.append(df_clean)

        # Load summary statistics
        mean_cof = np.loadtxt(self.base_path / f"{temp}" / "mean.txt")
        std_cof = np.loadtxt(self.base_path / f"{temp}" / "std.txt")

        # Verify consistency
        assert len(cycle_data) == len(mean_cof), f"Mismatch: {len(cycle_data)} cycles vs {len(mean_cof)} means"

        return {
            'data': cycle_data,
            'mean_cof': mean_cof,
            'std_cof': std_cof,
            'n_cycles': n_cycles
        }

    def create_pytorch_dataset(self, temps: list = [165, 167.5, 170]):
        """
        Create PyTorch-compatible dataset for all temperatures.

        Returns:
            dataset: List of dicts with keys:
                - 'M': transfer layer mass (to be initialized)
                - 'T': temperature
                - 'features': tensor [n_timesteps, 7] (position, velocity, forces, phase)
                - 'target_cof': tensor [n_timesteps] (observed COF)
                - 'cycle_num': int
                - 'temp': float
        """
        dataset = []

        for temp in temps:
            temp_data = self.load_all_cycles_for_temperature(temp)

            for cycle_idx, df in enumerate(temp_data['data']):
                # Extract features
                features = np.stack([
                    np.full(len(df), 0.5),  # M placeholder (will be updated during training)
                    np.full(len(df), (temp - 25) / (170 - 25)),  # T normalized
                    df['sliding_distance_norm'].values,
                    df['velocity_x'].values / 0.1,  # Normalize velocity (typical ~0.025 m/s)
                    df['force_x'].values / 10.0,  # Normalize forces (typical ~10 N)
                    df['force_y'].values / 10.0,
                    df['force_z'].values / 10.0,
                    df['cycle_phase'].values
                ], axis=1)

                target_cof = df['cof'].values

                dataset.append({
                    'M': 0.5,  # Initial placeholder
                    'T': temp,
                    'features': torch.tensor(features, dtype=torch.float32),
                    'target_cof': torch.tensor(target_cof, dtype=torch.float32),
                    'cycle_num': cycle_idx + 1,
                    'temp': temp,
                    'mean_cof_observed': temp_data['mean_cof'][cycle_idx]
                })

        return dataset
```

### 2.2 Dataset Statistics

After preprocessing, expected data volume:

| Temperature | Cycles | Avg Points/Cycle (after filter) | Total Points |
|-------------|--------|----------------------------------|--------------|
| 165°C | 114 | ~580 | ~66,000 |
| 167.5°C | 280 | ~580 | ~162,000 |
| 170°C | 142 | ~580 | ~82,000 |
| **Total** | **536** | **~580** | **~310,000** |

**Train/Val Split**: 80/20 per temperature (stratified)

---

## 3. Implementation Plan

### 3.1 Code Structure

```
GallingModel/
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── pinn_model.py            # Hybrid PINN architecture
│   ├── physics_module.py        # Physics equations for M(n) evolution
│   ├── train.py                 # Training loops (Stage 1 & 2)
│   ├── evaluate.py              # Validation and testing
│   └── visualization.py         # Plotting results
├── notebooks/
│   ├── 01_data_exploration.ipynb         # Explore 125Hz data
│   ├── 02_nn_pretraining.ipynb           # Stage 1 training
│   ├── 03_physics_informed_training.ipynb # Stage 2 joint training
│   └── 04_validation.ipynb               # Results and validation
├── config/
│   └── config.yaml              # Hyperparameters and settings
├── results/
│   ├── models/                  # Saved model checkpoints
│   ├── plots/                   # Validation plots
│   └── fitted_parameters.json   # Physics parameters
├── requirements.txt
└── main.py                       # CLI interface
```

### 3.2 Core PINN Model Implementation

**File**: `src/pinn_model.py`

```python
import torch
import torch.nn as nn
import numpy as np

class GallingPINN(nn.Module):
    """
    Physics-Informed Neural Network for galling prediction.

    Combines:
    1. Neural network for in-cycle COF(t) prediction
    2. Physics model for cycle-to-cycle M(n) evolution
    """

    def __init__(self, physics_params: dict):
        super().__init__()

        # Neural network component
        self.in_cycle_nn = InCycleCOF_NN()

        # Physics parameters (learnable)
        self.physics_params = nn.ParameterDict({
            'k0': nn.Parameter(torch.tensor(1.0)),
            'T0': nn.Parameter(torch.tensor(25.0)),
            'kw': nn.Parameter(torch.tensor(0.1)),
            'M0': nn.Parameter(torch.tensor(10.0)),
            'Tc': nn.Parameter(torch.tensor(45.0)),
            'alpha': nn.Parameter(torch.tensor(0.2))
        })

        # Fixed constants
        self.register_buffer('T_ref', torch.tensor(200.0))
        self.register_buffer('f_min', torch.tensor(0.15))
        self.register_buffer('f_max', torch.tensor(1.2))

    def forward_incycle(self, M, T, features):
        """
        Predict in-cycle COF using neural network.

        Args:
            M: Transfer layer mass (scalar or tensor)
            T: Temperature (scalar)
            features: [batch, time, 7] tensor with:
                      [M_norm, T_norm, sliding_dist, velocity, Fx, Fy, Fz, phase]

        Returns:
            cof_pred: [batch, time] predicted COF
        """
        # Update M in features
        features[..., 0] = M

        # Forward through NN
        # Reshape to [batch * time, 7] for NN
        batch_size, n_timesteps, _ = features.shape
        features_flat = features.reshape(-1, 7)

        cof_flat = self.in_cycle_nn(
            features_flat[:, 0],  # M_norm
            features_flat[:, 1],  # T_norm
            features_flat[:, 2],  # sliding_dist
            features_flat[:, 3],  # velocity
            features_flat[:, 4],  # Fx
            features_flat[:, 5],  # Fy
            features_flat[:, 6],  # Fz
            features_flat[:, 7]   # phase
        )

        cof_pred = cof_flat.reshape(batch_size, n_timesteps)
        return cof_pred

    def physics_update(self, M_current, T, mu_mean):
        """
        Physics-based mass update.

        Args:
            M_current: Current transfer layer mass
            T: Temperature (scalar)
            mu_mean: Mean COF during cycle

        Returns:
            M_next: Updated mass
        """
        # Attachment (Arrhenius)
        Q_attach = self.physics_params['k0'] * torch.exp(
            (T - self.T_ref) / self.physics_params['T0']
        )

        # Wear (friction-dependent)
        Q_wear = self.physics_params['kw'] * M_current * mu_mean

        # Mass update
        M_next = M_current + Q_attach - Q_wear

        # Critical mass threshold
        M_crit = self.physics_params['M0'] * torch.exp(
            -(T - self.T_ref) / self.physics_params['Tc']
        )

        # Detachment event
        if M_next >= M_crit:
            M_next = self.physics_params['alpha'] * M_next

        # Ensure non-negative
        M_next = torch.clamp(M_next, min=0.0)

        return M_next

    def forward_multi_cycle(self, T, features_list, M_init=0.0):
        """
        Simulate multiple cycles with coupled NN + physics.

        Args:
            T: Temperature (scalar)
            features_list: List of [time, 7] tensors (one per cycle)
            M_init: Initial transfer layer mass

        Returns:
            {
                'cof_predicted': List of [time] tensors (predicted COF per cycle),
                'M_history': [n_cycles] tensor (mass evolution),
                'mu_mean_history': [n_cycles] tensor (mean COF per cycle)
            }
        """
        n_cycles = len(features_list)
        M_current = M_init

        cof_predicted = []
        M_history = []
        mu_mean_history = []

        for cycle_idx, features in enumerate(features_list):
            # Add batch dimension
            features_batch = features.unsqueeze(0)  # [1, time, 7]

            # Predict in-cycle COF
            cof_pred = self.forward_incycle(M_current, T, features_batch)
            cof_pred = cof_pred.squeeze(0)  # [time]

            # Aggregate
            mu_mean = cof_pred.mean()

            # Physics update
            M_next = self.physics_update(M_current, T, mu_mean)

            # Store
            cof_predicted.append(cof_pred)
            M_history.append(M_current)
            mu_mean_history.append(mu_mean)

            # Update state
            M_current = M_next

        return {
            'cof_predicted': cof_predicted,
            'M_history': torch.stack(M_history),
            'mu_mean_history': torch.stack(mu_mean_history)
        }


class InCycleCOF_NN(nn.Module):
    """Neural network for within-cycle COF prediction"""

    def __init__(self, hidden_dims=[64, 128, 64, 32]):
        super().__init__()

        layers = []
        input_dim = 7  # M, T, x, v, Fx, Fy, Fz, phase

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self, M_norm, T_norm, sliding_dist, velocity, F_x, F_y, F_z, phase):
        """
        All inputs are 1D tensors of same length (n_timesteps)
        """
        x = torch.stack([M_norm, T_norm, sliding_dist, velocity,
                         F_x, F_y, F_z, phase], dim=-1)

        cof = self.network(x)
        cof = torch.sigmoid(cof) * 1.3  # Scale to [0, 1.3]

        return cof.squeeze(-1)
```

---

## 4. Success Criteria

### Minimum Success (Model is Valid)
- ✅ In-cycle RMSE < 0.10 for COF prediction
- ✅ Cycle-averaged RMSE < 0.15 vs mean.txt
- ✅ Qualitatively correct regimes (stable/oscillating/permanent)
- ✅ Physics parameters within expected ranges

### Good Success (Quantitatively Accurate)
- ✅ In-cycle RMSE < 0.05
- ✅ Cycle-averaged RMSE < 0.10
- ✅ Spike count within ±10%
- ✅ M(n) evolution shows realistic accumulation/detachment pattern
- ✅ Can predict intermediate temperatures (e.g., 168°C)

### Excellent Success (Publication-Quality)
- ✅ In-cycle RMSE < 0.03
- ✅ Cycle-averaged RMSE < 0.05
- ✅ Within-cycle spatial variation captured accurately
- ✅ Attach/detach events properly modeled
- ✅ Physics parameters physically interpretable (T0 ~ 20-30°C, Tc ~ 40-60°C)
- ✅ Extrapolates to 25°C (using only mean.txt for validation)

---

## 5. Implementation Tasks

1. ✅ Create project structure
2. ⏳ Implement `src/data_loader.py`
3. ⏳ Implement `src/pinn_model.py`
4. ⏳ Implement `src/train.py`
5. ⏳ Implement `src/evaluate.py`
6. ⏳ Implement `src/visualization.py`
7. ⏳ Create `config/config.yaml`
8. ⏳ Create `main.py`
9. ⏳ Create `requirements.txt`
10. ⏳ Create exploration notebook

---

## 6. Key Physics Parameters

| Parameter | Description | Expected Range | Initial Value |
|-----------|-------------|----------------|---------------|
| k0 | Baseline attachment rate | 0.001-10.0 | 1.0 |
| T0 | Attachment temp sensitivity | 10-50°C | 25.0 |
| kw | Wear rate constant | 0.01-0.5 | 0.1 |
| M0 | Baseline critical mass | 1.0-100.0 | 10.0 |
| Tc | Critical mass temp sensitivity | 20-80°C | 45.0 |
| alpha | Retention after detachment | 0.1-0.3 | 0.2 |

**Fixed Constants**:
- μ_min = 0.15
- μ_max = 1.2
- T_ref = 200°C

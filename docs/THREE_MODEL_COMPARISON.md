# Three-Model Architecture Comparison

This document describes the three-way comparison framework for evaluating different modeling approaches for galling prediction.

## Models

### 1. Feedforward PINN (Baseline)
- **File**: `src/pinn_model.py`
- **Description**: Neural network + physics hybrid
- **Parameters**: 19,271
- **Architecture**:
  - Feedforward NN: 8 inputs → [128, 64, 32] hidden layers → 1 output (COF)
  - Physics: Arrhenius attachment, friction-dependent wear, mass balance
- **Training**: Stage 1 (NN pre-training) + Stage 2 (joint optimization)
- **Strengths**: Learns complex temporal patterns from data
- **Reference Temperature**: 165°C

### 2. CNN-Hybrid PINN
- **File**: `src/pinn_model_cnn.py`
- **Description**: 1D CNN + physics hybrid
- **Parameters**: 7,479 (61% reduction from feedforward)
- **Architecture**:
  - 1D Conv layers: [16, 32] channels with kernel_size=5
  - FC layers: [64, 32]
  - Physics: Same equations as feedforward
- **Training**: Stage 1 (CNN pre-training) + Stage 2 (joint optimization)
- **Strengths**: Captures local temporal patterns, fewer parameters, parallelizable
- **Reference Temperature**: 165°C

### 3. Pure Physics-Only
- **File**: `src/pinn_model_physics_only.py`
- **Description**: Pure mechanistic model (no neural network)
- **Parameters**: 8 (physics parameters only)
- **Architecture**:
  - Friction law: μ = μ_base + μ_slope × M
  - Physics: Arrhenius attachment, friction-dependent wear, mass balance
- **Training**: Direct optimization of physics parameters
- **Strengths**: Minimal complexity, fully interpretable, no black-box component
- **Reference Temperature**: 165°C

## Comparison Framework

### Running the Comparison

```bash
# Run three-way comparison
python experiments/compare_all_three.py
```

This will:
1. Train all three models on the same dataset
2. Evaluate on the same test data
3. Compare metrics side-by-side
4. Save results to `results/comparison_all_three.json`

### Comparison Metrics

1. **Prediction Accuracy**
   - R² overall (goodness of fit)
   - RMSE in-cycle (timestep-level accuracy)
   - RMSE cycle-averaged (cycle-level accuracy)
   - Per-temperature performance (165°C, 167.5°C, 170°C)

2. **Model Complexity**
   - Total parameter count
   - Relative reduction vs feedforward baseline

3. **Computational Efficiency**
   - Total training time
   - Time ratio vs feedforward baseline

4. **Physics Parameters**
   - Fitted values for all physics parameters
   - Physical interpretability

## Key Questions Answered

### Q1: Does the neural network add value over pure physics?
Compare Feedforward/CNN vs Physics-Only:
- If NN models have significantly higher R², data-driven components are valuable
- If similar R², physics alone is sufficient

### Q2: Is CNN better than feedforward for this problem?
Compare CNN vs Feedforward:
- CNN has 61% fewer parameters
- Does it maintain or improve accuracy?
- Is training faster?

### Q3: What is the role of temporal patterns?
- Physics-only assumes constant COF during cycle (simple friction law)
- NN models learn complex in-cycle temporal dynamics
- Compare in-cycle RMSE to see value of temporal modeling

## Configuration Files

- `config/config.yaml` - Feedforward PINN
- `config/config_cnn.yaml` - CNN-Hybrid PINN
- `config/config_physics_only.yaml` - Pure Physics-Only

All use T_ref = 165°C for fair comparison.

## Output Structure

```
results/
├── comparison_all_three.json     # Three-way comparison results
├── models/                        # Feedforward model checkpoints
├── models_cnn/                    # CNN model checkpoints
├── models_physics/                # Physics-only model checkpoints
├── plots/                         # Feedforward plots
├── plots_cnn/                     # CNN plots
└── plots_physics/                 # Physics-only plots
```

## Expected Insights

1. **Value of Data-Driven Learning**:
   - NN models should capture complex patterns not in simplified physics
   - Improvement quantifies value of learning from data

2. **Architecture Trade-offs**:
   - CNN: fewer parameters, potentially faster, local pattern detection
   - Feedforward: more parameters, global pattern learning

3. **Physics vs Data Balance**:
   - Pure physics provides interpretable baseline
   - Hybrid models show improvement from data-driven corrections
   - If physics-only performs well, simpler model may be preferred

## Notes

- All models use identical physics equations (mass balance, Arrhenius, detachment)
- Only difference is how COF is predicted:
  - **Feedforward**: Neural network from all features
  - **CNN**: 1D convolution over temporal windows
  - **Physics-Only**: Simple friction law (μ = μ_base + μ_slope × M)

- Training approaches differ:
  - **Feedforward/CNN**: Two-stage (NN pre-train + joint optimization)
  - **Physics-Only**: Single-stage (direct physics optimization)

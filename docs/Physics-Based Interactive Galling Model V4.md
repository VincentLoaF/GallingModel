# Physics-Based Interactive Galling Model V4

## Abstract

**V4** removes `output_noise` entirely, making all variability emerge purely from **probabilistic competition**. A tiny fixed constant (`KDE_SMOOTHING = 0.005`) is used only for numerical stability in KDE density estimation.

### Changes from V3

| Aspect | V3 | V4 |
|--------|----|----|
| Output noise | Learnable `output_noise` (learned 0.0855) | **Removed entirely** |
| KDE stability | Via `output_noise` | Fixed constant `KDE_SMOOTHING = 0.005` |
| Variability source | Competition + residual noise | **Pure competition only** |
| Parameter count | 14 learnable | **13 learnable** (removed `output_noise`) |
| LOW state σ | ~0.109 (too high — noise contributes 54%) | Expected ~0.037 (competition-only) |

**Motivation**: V3 analysis revealed that `output_noise` learned to 0.0855 (2.3x training target σ=0.037), contributing 54% of LOW state variance. This masks the competition mechanism and inflates oscillations where the data is stable. Removing it forces the optimizer to rely entirely on the probabilistic competition dynamics.

---

## 1. Core Concept: Pure Probabilistic Competition

Identical to V3 — each cycle is a **discrete competition** between:
- **Accumulation**: Tries to add mass (galling progresses)
- **Healing**: Tries to remove mass (surface recovers)

The winner is determined probabilistically, with odds depending on:
1. **Physical rates** — thermodynamic driving force
2. **Current state ($\beta$)** — regime awareness

**V4 difference**: There is **no output noise layer**. The friction output is deterministic given the state. All stochasticity comes from the competition outcomes.

---

## 2. Governing Equations

### Equations A–C: Identical to V3

- **Equation A**: Growth probability $P_{growth} = P_{base} \cdot f(\beta)$
- **Equation B**: Discrete mass update (growth step OR removal step)
- **Equation C**: State mapping $\rho_n = \tanh(M_n / M_{sat})$

See V3 documentation for full details.

### Equation D: Friction Output (V4 Change)

$$
\mu_{next} = (1 - \beta_n) \cdot \mu_{clean} + \beta_n \cdot \mu_{galled} + \mathcal{N}(0, \sigma_{KDE}^2)
$$

Where $\sigma_{KDE} = 0.005$ is a **fixed constant** (not learnable), used only for KDE numerical stability during training. This is not a physics parameter — it prevents degenerate density estimates when simulated values cluster tightly.

**V4 Change**: Removed `output_noise` as learnable parameter. The $\sigma_{KDE} = 0.005$ term is negligible compared to competition-driven variability.

### Equation E: Massive Detachment Events

Identical to V3.

---

## 3. How It Explains Bistability

Same mechanism as V3, but now **all variability** must come from competition:

### Why LOW State Should Be More Stable in V4

- V3: $\sigma_{LOW} \approx 0.109$ (competition σ=0.07 + noise σ=0.085)
- V4: $\sigma_{LOW}$ should approach training target (~0.037)
- Without the noise crutch, the optimizer must tune competition parameters to match observed stability

### Why HIGH State Oscillates

Same as V3 — balanced competition at $\beta \approx 1$ creates genuine oscillations.

### Why TRANSITION is Chaotic

Same as V3 — transition boost maximizes uncertainty at $\beta \approx 0.5$.

---

## 4. Summary of Variables

### State Variables

| Symbol | Meaning | Role |
|--------|---------|------|
| $T$ | Temperature | Controls growth/healing balance |
| $\mu$ | Friction Coefficient | Output and feedback driver |
| $M$ | Transfer Mass | Internal memory |
| $\rho$ | Galling Density | Surface coverage ($0 \le \rho \le 1$) |
| $\beta$ | Regime Parameter | Determines competition dynamics |
| $P_{growth}$ | Growth Probability | Chance that accumulation wins this cycle |

### Learnable Physics Parameters (13 total)

| Symbol | Code Name | Initial | Meaning |
|--------|-----------|---------|---------|
| $k_{adh}$ | `k_adh` | 0.05 | Base adhesion rate |
| $\alpha$ | `alpha_tau` | 3.5 | Shear sensitivity (feedback strength) |
| $k_{spall}$ | `k_spall` | 0.8 | Base spalling/healing rate |
| $c_2$ | `E_spall` | 0.5 | Thermal activation energy |
| $M_{sat}$ | `M_sat` | 5.0 | Saturation mass threshold |
| $\rho_{crit}$ | `rho_crit` | 0.4 | Critical density for regime shift |
| $k$ | `beta_sharpness` | 15.0 | Transition sharpness |
| $\gamma$ | `beta_influence` | 2.0 | Nonlinearity of $\beta$ effect on P(growth) |
| $s_{low}$ | `low_state_suppression` | 0.1 | Growth suppression at clean state |
| $\kappa$ | `transition_boost` | 1.5 | Extra randomness at transition |
| $\Delta_{growth}$ | `growth_step` | 0.3 | Mass gain when growth wins |
| $\Delta_{removal}$ | `removal_step` | 0.2 | Mass loss rate when healing wins |
| $p_0$ | `prob_detach` | 0.03 | Base detachment probability |

### Fixed Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| $T_{ref}$ | 165°C | Reference temperature |
| $\mu_{clean}$ | 0.15 | Clean surface friction |
| $\mu_{galled}$ | 1.0 | Galled surface friction |
| $c_1$ | 0.2 | Thermal softening coefficient |
| $\sigma_{KDE}$ | 0.005 | KDE numerical stability (NOT physics) |

---

## 5. Implementation Notes

### Code Structure

Model file: `src/models/interactive_galling_model_v4.py`

1. **`compute_growth_probability(T, μ, M, β)`**: Calculates P(growth wins) based on physics + state
2. **`update_state(M, ρ, T, μ)`**: Draws random outcome, applies discrete step
3. **`compute_friction(ρ)`**: Computes friction with fixed KDE smoothing only (no learnable noise)
4. **`simulate_multiple_cycles(T, n_cycles)`**: Runs simulation, tracks outcomes

### Training

```bash
python scripts/train_interactive_galling_v4.py                    # Train from scratch
python scripts/train_interactive_galling_v4.py --resume-from-best # Fine-tune
python scripts/train_interactive_galling_v4.py --validate-only    # Validation only
```

### Key V4 Innovations

1. **No Output Noise**: All variability from probabilistic competition only
2. **Pure Competition Dynamics**: Optimizer must tune competition parameters to match data
3. **Fixed KDE Smoothing**: 0.005 constant for numerical stability, not physics
4. **Fewer Parameters**: 13 learnable (removed `output_noise` from V3's 14)

### Training Results

**Training:** 133 epochs (early stopping), best model at epoch 102.

| Metric | Value |
|--------|-------|
| Best LL | -468.90 (epoch 102) |
| Final LL | -543.25 (epoch 133) |
| Initial LL | -5332.02 |
| Comparison | Better than V3 (-479.57) at best epoch |

**Per-Temperature Performance (best epoch):**

| Temperature | LL | Mean ρ | Final ρ | High Cycles | 1st Transition |
|-------------|-----|--------|---------|-------------|----------------|
| 165°C | -66.79 | 0.036 | 0.032 | 0 | None (stable LOW) |
| 167.5°C | -287.86 | 0.230 | 0.367 | 160 | Cycle 108 |
| 170°C | -114.26 | 0.376 | 0.725 | 76 | Cycle 66 |

**Learned Parameters:**

| Parameter | Initial | Learned | Notes |
|-----------|---------|---------|-------|
| `growth_step` | 0.3 | 0.713 | 2.4× larger than init |
| `removal_step` | 0.2 | 0.007 | Nearly zero — healing disabled |
| `M_sat` | 5.0 | 4.544 | Close to initial |
| `rho_crit` | 0.4 | 0.186 | Shifted lower — earlier transition |
| `beta_sharpness` | 15.0 | 14.94 | Unchanged |

**Assessment:** V4 marginally improved over V3 (-469 vs -480), with better 165°C and 170°C fits. However, `removal_step` collapsed to near zero, meaning the healing mechanism was effectively disabled. The pure competition model without output noise did not improve LOW state stability as hoped — the optimizer found a degenerate solution where growth dominates and healing is suppressed.

### Expected Behavior

| Temperature | Dominant State | P(growth) | Stability |
|-------------|----------------|-----------|-----------|
| 165°C | Clean (LOW) | ~5% | Very stable (σ ≈ 0.037 target) |
| 167.5°C | Bistable | ~20-50% | Chaotic |
| 170°C | Galled (HIGH) | ~50% | Oscillating |

### Training Data Match (Expected)

| Region | Training Data σ | V4 Mechanism |
|--------|-----------------|--------------|
| LOW ($\mu < 0.25$) | 0.02-0.04 | P(growth) ≈ 5% → healing dominates, no noise inflation |
| MID (0.25-0.6) | 0.08-0.11 | Transition boost → uncertain competition |
| HIGH ($\mu \geq 0.6$) | 0.13-0.18 | P(growth) ≈ 50% → active competition |

---

# Physics-Based Interactive Galling Model V5

## Abstract

**V5** replaces the probabilistic competition mechanism (V3/V4) with a physically motivated **growth + detachment** model. Mass accumulates via stochastic growth events and is removed by stochastic detachment events, with a small passive decay providing continuous healing.

### Changes from V4

| Aspect | V4 (Probabilistic Competition) | V5 (Growth + Detachment) |
|--------|-------------------------------|--------------------------|
| Growth mechanism | Binary competition (growth vs healing) | Stochastic events with state-dependent rate |
| Removal mechanism | Healing wins the competition | Stochastic detachment (chunk removal) |
| Variability source | Which side wins the coin flip | Detachment timing and severity |
| Growth rate | Fixed step size when growth wins | Peaks at MID β, lower at LOW and HIGH |
| Removal severity | Fixed step (removal_step × M) | Variable: Uniform(f_min, f_max) |
| Passive healing | None (only competition) | Small continuous decay (λ × M) |
| Detachment trigger | Fixed probability | Mass + temperature dependent |
| Parameter count | 13 learnable | 14 learnable |
| New parameters | — | `p_growth_base`, `s_peak`, `s_base`, `decay_rate`, `p_detach_base`, `M_half`, `c_stick`, `f_min`, `f_max` |
| Removed parameters | — | `k_spall`, `E_spall`, `beta_influence`, `low_state_suppression`, `transition_boost`, `growth_step`, `removal_step`, `prob_detach` |

**Motivation**: V4 analysis revealed that the discrete competition mechanism produces binary mass jumps (saw-tooth patterns), not the smooth oscillations seen in training data. The growth + detachment model better captures the physics: aluminum transfer builds up gradually during sliding and flakes off in chunks.

---

## 1. Core Concept: Growth and Detachment

The model treats each cycle as having three independent processes:

1. **Stochastic Growth**: Material transfer occurs probabilistically. When it happens, the rate depends on the current galling state — highest in the active galling zone (MID β), lower when clean or saturated.

2. **Passive Decay**: A small amount of material is continuously removed each cycle, representing gradual healing processes.

3. **Stochastic Detachment**: Accumulated material can suddenly flake off. The probability increases with mass (thicker buildup is more fragile) and decreases with temperature (hotter material is stickier). The severity varies — each event removes a different fraction of mass.

This naturally captures the key observation from training data:
- **LOW state ($\mu < 0.25$)**: Growth events are rare (low T) → decay keeps mass near zero → **very stable** ($\sigma \approx 0.03$)
- **HIGH state ($\mu \geq 0.6$)**: Growth is active → mass accumulates → detachments create oscillations ($\sigma \approx 0.15$)
- **Transition**: Growth pushes mass up → detachments knock it down → **chaotic bistability**

---

## 2. Governing Equations

### Equation A: Growth Event

**Step 1 — Does growth occur this cycle?**

$$
P_{growth} = \text{clamp}\left(\frac{p_{g,base} \cdot \Theta(T)}{1 + \Theta(T)}, 0.01, 0.99\right)
$$

where $\Theta(T) = e^{0.2(T - T_{ref})}$ is thermal softening. Higher temperature increases growth probability.

**Step 2 — If growth occurs, how much?**

$$
\Delta M_{growth} = k_{adh} \cdot \Theta(T) \cdot \mu^{\alpha_\tau} \cdot S(\beta)
$$

**State-dependent scaling:**

$$
S(\beta) = 4\beta(1-\beta) \cdot s_{peak} + s_{base}
$$

| $\beta$ | $S(\beta)$ | Physical meaning |
|---------|------------|------------------|
| ≈ 0 (clean) | $s_{base}$ (small) | Little adhesion opportunity |
| ≈ 0.5 (active galling) | $s_{peak} + s_{base}$ (maximum) | Peak material transfer zone |
| ≈ 1 (saturated) | $s_{base}$ (small) | Near saturation, less room |

### Equation B: Passive Decay

$$
\Delta M_{decay} = -\lambda \cdot M
$$

where $\lambda$ is a small learnable decay rate. This provides gentle drift toward the clean state when growth is absent.

### Equation C: Detachment Event

**Probability (mass + temperature dependent):**

$$
P_{detach} = \text{clamp}\left(p_0 \cdot \frac{M}{M + M_{half}} \cdot e^{-c_{stick} \cdot \Delta T}, 0, 0.95\right)
$$

- $M / (M + M_{half})$: Saturating — more mass → more likely to flake
- $e^{-c_{stick} \cdot \Delta T}$: Hotter → stickier → less detachment

### Equation D: Detachment Severity

When detachment occurs, the fraction of mass removed is drawn from:

$$
f_{severity} \sim \text{Uniform}(f_{min}, f_{max})
$$

$$
M_{after} = M_{before} \cdot (1 - f_{severity})
$$

### Equation E: Net Mass Update

$$
M_{next} = \begin{cases}
(M + \Delta M_{growth} + \Delta M_{decay}) \cdot (1 - f_{severity}) & \text{if detachment occurs} \\
M + \Delta M_{growth} + \Delta M_{decay} & \text{otherwise}
\end{cases}
$$

$$
M_{next} = \max(M_{next}, 0)
$$

### Equation F: State Mapping (Mass → Density)

$$
\rho_n = \tanh\left(\frac{M_n}{M_{sat}}\right)
$$

### Equation G: Friction Output (Density → Friction)

$$
\mu_{next} = (1 - \beta_n) \cdot \mu_{clean} + \beta_n \cdot \mu_{galled} + \mathcal{N}(0, \sigma_{KDE}^2)
$$

where $\beta_n = \sigma(k(\rho_n - \rho_{crit}))$ and $\sigma_{KDE} = 0.005$ is fixed (KDE stability only).

---

## 3. How It Explains Bistability

### Why LOW State is Stable

At low temperature (165°C):
- $\Theta(T) \approx 1$ → $P_{growth} \approx 0.3$ (moderate)
- But growth magnitude is small (low $\mu$, low $S(\beta)$)
- Decay ($\lambda M$) keeps mass near zero
- Detachment is irrelevant (no mass to detach)
- Result: Mass stays low → COF stays at $\mu_{clean} \approx 0.15$

### Why HIGH State Oscillates

At high temperature (170°C):
- $\Theta(T) \approx 2.7$ → $P_{growth} \approx 0.5$ (frequent)
- Growth magnitude is large (high $\mu$, high $\Theta$)
- Mass accumulates → detachment probability increases
- Frequent detachments with variable severity create irregular oscillations
- Result: Mass oscillates → COF oscillates around $0.6 - 1.0$

### Why TRANSITION is Chaotic

At critical temperature (167.5°C):
- Growth can push system into galled state
- Detachments can knock it back to clean state
- The system jumps between LOW and HIGH unpredictably
- Result: Bistable behavior with chaotic transitions

### Temperature Effects

| Temperature | $\Theta(T)$ | $P_{growth}$ | Detachment | Expected Behavior |
|-------------|-------------|--------------|------------|-------------------|
| 165°C (Low) | ~1.0 | ~30% | Rare | Decay dominates → stable LOW |
| 167.5°C (Critical) | ~1.6 | ~40% | Moderate | Bistable → either regime |
| 170°C (High) | ~2.7 | ~50% | Frequent (but sticky) | Growth wins → oscillating HIGH |

---

## 4. Summary of Variables

### State Variables

| Symbol | Meaning | Role |
|--------|---------|------|
| $T$ | Temperature | Controls growth rate and detachment |
| $\mu$ | Friction Coefficient | Output and feedback driver |
| $M$ | Transfer Mass | Internal memory (accumulated material) |
| $\rho$ | Galling Density | Surface coverage ($0 \le \rho \le 1$) |
| $\beta$ | Regime Parameter | Sigmoid of density, determines COF |

### Learnable Physics Parameters (14 total)

| Symbol | Code Name | Initial | Meaning |
|--------|-----------|---------|---------|
| $k_{adh}$ | `k_adh` | 0.05 | Base adhesion rate |
| $\alpha_\tau$ | `alpha_tau` | 3.5 | Shear sensitivity (friction feedback) |
| $p_{g,base}$ | `p_growth_base` | 0.6 | Base probability of growth event |
| $s_{peak}$ | `s_peak` | 1.5 | Growth scaling peak (at β≈0.5) |
| $s_{base}$ | `s_base` | 0.3 | Baseline growth scaling |
| $\lambda$ | `decay_rate` | 0.02 | Passive decay rate |
| $p_0$ | `p_detach_base` | 0.08 | Base detachment probability |
| $M_{half}$ | `M_half` | 1.0 | Mass at 50% detachment prob |
| $c_{stick}$ | `c_stick` | 0.3 | Temperature stickiness factor |
| $f_{min}$ | `f_min` | 0.3 | Minimum detachment severity |
| $f_{max}$ | `f_max` | 0.8 | Maximum detachment severity |
| $M_{sat}$ | `M_sat` | 5.0 | Saturation mass |
| $\rho_{crit}$ | `rho_crit` | 0.4 | Critical density threshold |
| $k$ | `beta_sharpness` | 15.0 | Transition sharpness |

### Fixed Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| $T_{ref}$ | 165°C | Reference temperature |
| $\mu_{clean}$ | 0.15 | Clean surface friction |
| $\mu_{galled}$ | 1.0 | Galled surface friction |
| $\sigma_{KDE}$ | 0.005 | KDE numerical stability (NOT physics) |

---

## 5. Implementation Notes

### Code Structure

Model file: `src/models/interactive_galling_model_v5.py`

1. **`update_state(M, ρ, T, μ)`**: Three-process update (growth + decay + detachment)
2. **`compute_friction(ρ)`**: Sigmoid beta mapping with fixed KDE smoothing
3. **`simulate_multiple_cycles(T, n_cycles)`**: Runs simulation, tracks all events
4. **`get_physics_params()`**: Returns dict of learned parameter values

### Training

```bash
python scripts/train_interactive_galling_v5.py                    # Train from scratch
python scripts/train_interactive_galling_v5.py --resume-from-best # Fine-tune
python scripts/train_interactive_galling_v5.py --validate-only    # Validation only
```

### Key V5 Innovations

1. **Stochastic Growth**: Events occur probabilistically, rate peaks at MID β
2. **Variable Detachment**: Mass+T dependent probability, Uniform severity
3. **Passive Decay**: Continuous healing drift
4. **No Output Noise**: All variability from growth/detachment dynamics
5. **Physically Motivated**: Aluminum transfer builds up and flakes off in chunks

### Training Results

**Training:** 165 epochs (early stopping), best model at epoch 134.

| Metric | Value |
|--------|-------|
| Best LL | -1000.44 (epoch 134) |
| Final LL | -1135.87 (epoch 165) |
| Initial LL | -5332.37 |
| Comparison | 2× worse than V3 (-479.57) and V4 (-468.90) |

**Per-Temperature Performance (best epoch):**

| Temperature | LL | Mean ρ | Final ρ | High Cycles | 1st Transition |
|-------------|-----|--------|---------|-------------|----------------|
| 165°C | -125.34 | 0.009 | 0.011 | 0 | None (stable LOW) |
| 167.5°C | -703.85 | 0.232 | 0.434 | 149 | Cycle 128 |
| 170°C | -171.25 | 0.557 | 0.764 | 111 | Cycle 31 |

**Learned Parameters:**

| Parameter | Initial | Learned | Notes |
|-----------|---------|---------|-------|
| `k_adh` | 0.05 | 0.435 | 8.7× larger — aggressive growth |
| `s_peak` | 1.5 | 1.82 | Slightly above initial |
| `decay_rate` | 0.02 | 0.062 | 3× larger — strong passive decay |
| `M_sat` | 5.0 | 4.607 | Close to initial |
| `rho_crit` | 0.4 | 0.139 | Shifted very low |
| `beta_sharpness` | 15.0 | 14.77 | Unchanged |
| `f_min` | 0.3 | 0.223 | Mild detachment |
| `f_max` | 0.8 | 0.740 | Large detachment chunks |

**Assessment:** V5 significantly underperformed all previous versions. Key failures:
- **167.5°C catastrophic** (LL=-704): The growth+detachment mechanism failed to capture bistability. Cold-start behavior (late first transition at cycle 128) indicates the model can't spontaneously enter the HIGH state early enough.
- **LOW state too stable**: Mean ρ=0.009 at 165°C vs expected ~0.04 — almost no activity at all, producing σ≈0.007 instead of target σ≈0.037.
- **Parameter compensation**: Large `k_adh` (0.435) offset by large `decay_rate` (0.062), suggesting the growth and decay processes fight each other rather than cooperating.

The growth+detachment model was more physically motivated but proved harder to optimize than the simpler competition mechanism.

### Expected Behavior

| Temperature | Dominant State | Growth Events | Detachment Events | Stability |
|-------------|----------------|---------------|-------------------|-----------|
| 165°C | Clean (LOW) | Rare, small | Negligible | Very stable |
| 167.5°C | Bistable | Moderate | Moderate | Chaotic |
| 170°C | Galled (HIGH) | Frequent, large | Frequent (but sticky) | Oscillating |

### Training Data Match (Expected)

| Region | Training Data σ | V5 Mechanism |
|--------|-----------------|--------------|
| LOW ($\mu < 0.25$) | 0.02-0.04 | Growth rare + decay → mass stays near zero |
| MID (0.25-0.6) | 0.08-0.11 | Transition zone → unpredictable |
| HIGH ($\mu \geq 0.6$) | 0.13-0.18 | Growth active + detachment → irregular oscillations |

---

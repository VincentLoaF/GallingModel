# Physics-Based Interactive Galling Model V1

## Abstract

**V1** extends V0 by adding **state-dependent output noise** to capture different stability levels across regimes.

### Changes from V0

| Aspect | V0 | V1 |
|--------|----|----|
| Output noise | Constant $\sigma$ | State-dependent $\sigma(\beta)$ |
| Parameters | 9 learnable | 12 learnable (+3 noise params) |

### Key Addition

$$
\sigma(\beta) = \sigma_{base} + A \cdot \beta + B \cdot \beta(1-\beta)
$$

This creates a stability ranking: $\sigma_{clean} < \sigma_{galled} < \sigma_{transition}$

**Motivation**: Training data shows the LOW state (μ ≈ 0.15) is much more stable than the HIGH state (μ ≈ 1.0), which V0's constant noise cannot capture.

---

## 1. Core Concept: A Feedback Loop

The model relies on a **closed-loop feedback mechanism**. Unlike simple models where temperature alone dictates wear, this model acknowledges that **friction causes galling, and galling causes friction**.

The system is defined by two internal state variables that evolve cycle-by-cycle ($n$):

- $M_n$: **Transfer Mass** — The amount of aluminum stuck to the die
- $\rho_n$: **Galling Density** — The fractional surface area covered by aluminum ($0 \le \rho \le 1$)

---

## 2. Governing Equations

### Equation A: Mass Evolution (The Competition)

The change in transfer mass ($\Delta M$) is the net result of **Adhesive Growth** competing against **Spalling (Self-Healing)**.

$$
\Delta M = \underbrace{k_{adh} \cdot \Theta(T) \cdot (\mu_n)^{\alpha}}_{\text{Growth Term}} - \underbrace{k_{spall} \cdot \Phi(T) \cdot M_n}_{\text{Healing Term}} + \eta
$$

**Where:**

- $\mu_n$: The current Coefficient of Friction
- $(\mu_n)^{\alpha}$: **The Feedback Factor** — Since $\alpha > 1$, a small increase in friction causes a massive increase in growth rate
- $\eta$: Process noise $\sim \mathcal{N}(0, \sigma_{noise}^2)$
- $\Theta(T)$: Thermal Softening function
  $$\Theta(T) = \exp\left( c_1 \cdot (T - T_{ref}) \right)$$
- $\Phi(T)$: Thermal Stickiness function (healing suppression)
  $$\Phi(T) = \exp\left( -c_2 \cdot (T - T_{ref}) \right)$$

### Equation B: State Mapping (Mass → Density)

As mass accumulates, it spreads to cover the surface with saturation behavior:

$$
\rho_n = \tanh\left( \frac{M_n}{M_{sat}} \right)
$$

### Equation C: Friction Output (Density → Friction)

The friction coefficient is determined by a mixing rule:

$$
\mu_{next} = (1 - \beta_n) \cdot \mu_{clean} + \beta_n \cdot \mu_{galled}
$$

Where $\beta_n$ is the **Regime Parameter** (sigmoid transition):

$$
\beta_n = \frac{1}{1 + e^{-k(\rho_n - \rho_{crit})}}
$$

### Equation D: State-Dependent Noise (V1 Addition)

The friction output includes noise that varies based on the current regime:

$$
\mu_{output} = \mu_{next} + \mathcal{N}(0, \sigma(\beta)^2)
$$

Where the noise standard deviation is a function of the regime parameter:

$$
\sigma(\beta) = \sigma_{base} + A \cdot \beta + B \cdot \beta(1-\beta)
$$

**Physical Interpretation:**

| State | $\beta$ | $\sigma$ | Physical Reason |
|-------|---------|----------|-----------------|
| Clean | ≈ 0 | $\sigma_{base}$ (Lowest) | Smooth steel-Al interface |
| Galled | ≈ 1 | $\sigma_{base} + A$ (Medium) | Rough but stable Al-on-Al |
| Transition | ≈ 0.5 | Maximum | Active tearing/re-attachment |

**Constraint**: For stability ranking $\sigma_{clean} < \sigma_{galled} < \sigma_{transition}$, require $B > 2A$.

### Equation E: Massive Detachment Events

Probabilistic event where large chunks break off, acting as a restoring force:

$$
P_{detach} = p_0 \cdot \exp\left( -0.5 \cdot (T - T_{ref}) \right)
$$

If triggered: $M_{next} = 0.5 \cdot M_{next}$

---

## 3. How It Explains Bistability

### Scenario 1: Low Temperature (165°C)

- **Physics**: Aluminum is harder ($\Theta(T)$ low) and less sticky ($\Phi(T)$ high)
- **Behavior**: Growth term is tiny, healing dominates
- **Noise**: $\sigma \approx \sigma_{base}$ (lowest) — very stable
- **Result**: System trapped in **"Clean Well"** with minimal oscillation

### Scenario 2: High Temperature (170°C)

- **Physics**: Aluminum softens ($\Theta(T)$ high) and becomes sticky ($\Phi(T)$ low)
- **Behavior**: Runaway feedback leads to permanent galling
- **Noise**: $\sigma \approx \sigma_{base} + A$ (medium) — moderate oscillation around μ ≈ 1.0
- **Result**: System in **"Galled Well"** with visible variability

### Scenario 3: Transitional Temperature (167.5°C)

- **Physics**: Growth and healing rates are nearly balanced
- **Behavior**: System can exist in either regime
- **Noise**: $\sigma$ peaks at transition — maximum instability
- **Result**: **Meta-stable behavior** with highest cycle-to-cycle variability

---

## 4. Summary of Variables

### State Variables

| Symbol | Meaning | Role |
|--------|---------|------|
| $T$ | Temperature | Control parameter |
| $\mu$ | Friction Coefficient | Output and feedback driver |
| $M$ | Transfer Mass | Internal memory |
| $\rho$ | Galling Density | Surface coverage ($0 \le \rho \le 1$) |
| $\beta$ | Regime Parameter | Transition probability |

### Learnable Physics Parameters

| Symbol | Code Name | Learned | Meaning |
|--------|-----------|---------|---------|
| $k_{adh}$ | `k_adh` | 0.16 | Base adhesion rate |
| $\alpha$ | `alpha_tau` | 3.34 | Shear sensitivity (feedback strength) |
| $k_{spall}$ | `k_spall` | 0.64 | Base spalling/healing rate |
| $c_2$ | `E_spall` | 0.65 | Thermal activation energy |
| $M_{sat}$ | `M_sat` | 4.82 | Saturation mass threshold |
| $\rho_{crit}$ | `rho_crit` | 0.22 | Critical density for regime shift |
| $k$ | `beta_sharpness` | 14.83 | Transition sharpness |
| — | `noise_lvl` | 0.34 | Process noise amplitude |
| $p_0$ | `prob_detach` | 0.05 | Base detachment probability |
| $\sigma_{base}$ | `sigma_base` | 0.15 | Baseline output noise (V1) |
| $A$ | `sigma_galled` | 0.03 | Additional noise at galled (V1) |
| $B$ | `sigma_transition` | 0.36 | Transition instability peak (V1) |

### Fixed Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| $T_{ref}$ | 165°C | Reference temperature |
| $\mu_{clean}$ | 0.15 | Clean surface friction |
| $\mu_{galled}$ | 1.0 | Galled surface friction |
| $c_1$ | 0.2 | Thermal softening coefficient |

---

## 5. Noise at Different Regimes

With the learned parameters:

| Regime | $\beta$ | $\sigma$ | Interpretation |
|--------|---------|----------|----------------|
| Clean | 0 | 0.146 | Lowest — predictable steel-Al contact |
| Transition | 0.5 | 0.248 | Highest — active surface changes |
| Galled | 1 | 0.173 | Medium — rough but stable Al-on-Al |

---

## 6. Implementation Notes

### Code Structure

Model file: `src/models/interactive_galling_model.py`

1. **`compute_rates(T, μ)`**: Calculates growth and removal rates
2. **`update_state(M, ρ, T, μ)`**: Evolves mass and density, handles detachment
3. **`compute_friction(ρ)`**: Computes friction with **state-dependent noise**
4. **`simulate_multiple_cycles(T, n_cycles)`**: Runs full simulation

### Training

```bash
python scripts/train_interactive_galling.py                    # Train from scratch
python scripts/train_interactive_galling.py --resume-from-best # Fine-tune
python scripts/train_interactive_galling.py --validate-only    # Validation only
```

### Key V1 Innovations

1. **State-Dependent Noise**: $\sigma(\beta)$ captures stability ranking
2. **Three noise parameters**: Separate control for clean, galled, and transition states
3. **Physical interpretation**: Different surface states have different variability

### Training Results

**Best Log-Likelihood**: -534.10 at epoch 49/79

| Temperature | Log-Likelihood | Mean β | High Cycles |
|-------------|----------------|--------|-------------|
| 165°C | -111.79 | 0.078 | 0 |
| 167.5°C | -297.25 | 0.361 | 7 |
| 170°C | -125.06 | 0.833 | 127 |

**Key Learned Parameters:**

| Parameter | Learned Value | Physical Meaning |
|-----------|---------------|------------------|
| `sigma_base` | 0.1458 | Baseline output noise |
| `sigma_galled` | 0.0272 | Additional noise at galled state |
| `sigma_transition` | 0.3552 | Transition instability peak |
| `rho_crit` | 0.2189 | Critical density threshold |
| `k_adh` | 0.1606 | Adhesion rate |
| `k_spall` | 0.6432 | Spalling rate |

**Observations:**
- Successfully captures stability ranking: 165°C (stable) → 170°C (galled)
- 167.5°C has highest uncertainty (lowest LL per data point) due to transitional behavior
- Noise parameters achieve desired ranking: $\sigma_{clean} < \sigma_{galled} < \sigma_{transition}$

### Limitations

- Noise is a **phenomenological fit** rather than emerging from physics
- Does not explain **why** stability differs across states
- Oscillations at HIGH state come from noise, not dynamics

---

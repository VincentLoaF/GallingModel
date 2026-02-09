# Physics-Based Interactive Galling Model V2

## Abstract

**V2** attempts to explain oscillations through **physics-based mechanisms** rather than state-dependent noise, introducing:

1. **Non-linear healing**: Removal scales as $M^n$ where $n > 1$
2. **Competition intensity**: Dynamics amplified at transition ($\beta \approx 0.5$)

### Changes from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Output noise | State-dependent $\sigma(\beta)$ | Constant $\sigma_{output}$ |
| Healing term | Linear: $k \cdot M$ | Non-linear: $k \cdot M^n$ |
| Transition dynamics | Normal | Amplified by $I(\beta)$ |
| New parameters | — | `healing_exponent`, `competition_kappa` |
| Removed parameters | — | `sigma_base`, `sigma_galled`, `sigma_transition` |

**Motivation**: V1's state-dependent noise is phenomenological. V2 attempts to make oscillations emerge from physics: thick layers ($M^n$) are mechanically unstable, and transition zones have aggressive competition.

---

## 1. Core Concept: A Feedback Loop

The model relies on a **closed-loop feedback mechanism**. Unlike simple models where temperature alone dictates wear, this model acknowledges that **friction causes galling, and galling causes friction**.

The system is defined by two internal state variables that evolve cycle-by-cycle ($n$):

- $M_n$: **Transfer Mass** — The amount of aluminum stuck to the die
- $\rho_n$: **Galling Density** — The fractional surface area covered by aluminum ($0 \le \rho \le 1$)

---

## 2. Governing Equations

### Equation A: Mass Evolution with Non-Linear Healing

$$
\Delta M = I(\beta) \cdot \left[ \underbrace{k_{adh} \cdot \Theta(T) \cdot \mu^{\alpha}}_{\text{Growth Term}} - \underbrace{k_{spall} \cdot \Phi(T) \cdot M^n}_{\text{Non-linear Healing}} \right] + \eta
$$

**V2 Changes:**

| Term | V0/V1 | V2 |
|------|-------|----|
| Healing | $k_{spall} \cdot \Phi(T) \cdot M$ | $k_{spall} \cdot \Phi(T) \cdot M^n$ |
| Dynamics | Fixed rate | Amplified by $I(\beta)$ |

**Where:**

- $n$: Healing exponent — when $n > 1$, thick layers experience super-linear removal
- $I(\beta)$: Competition intensity — amplifies both growth and removal at transition
- $\eta$: Process noise $\sim \mathcal{N}(0, \sigma_{noise}^2)$
- $\Theta(T)$: Thermal Softening
  $$\Theta(T) = \exp\left( c_1 \cdot (T - T_{ref}) \right)$$
- $\Phi(T)$: Thermal Stickiness
  $$\Phi(T) = \exp\left( -c_2 \cdot (T - T_{ref}) \right)$$

### Equation B: Competition Intensity (V2 Addition)

$$
I(\beta) = 1 + \kappa \cdot \beta(1-\beta)
$$

| $\beta$ | $I(\beta)$ | Interpretation |
|---------|------------|----------------|
| 0 (clean) | 1.0 | Normal dynamics |
| 0.5 (transition) | $1 + \kappa/4$ | Maximum amplification |
| 1 (galled) | 1.0 | Normal dynamics |

**Physical Interpretation**: During transition, neither state dominates — both growth and removal are aggressive, creating instability.

### Equation C: State Mapping (Mass → Density)

$$
\rho_n = \tanh\left( \frac{M_n}{M_{sat}} \right)
$$

### Equation D: Friction Output with Constant Noise

$$
\mu_{next} = (1 - \beta_n) \cdot \mu_{clean} + \beta_n \cdot \mu_{galled} + \mathcal{N}(0, \sigma_{output}^2)
$$

Where $\beta_n$ is the **Regime Parameter**:

$$
\beta_n = \frac{1}{1 + e^{-k(\rho_n - \rho_{crit})}}
$$

**V2 Change**: $\sigma_{output}$ is constant. Oscillations should emerge from non-linear healing dynamics.

### Equation E: Massive Detachment Events

$$
P_{detach} = p_0 \cdot \exp\left( -0.5 \cdot (T - T_{ref}) \right)
$$

If triggered: $M_{next} = 0.5 \cdot M_{next}$

---

## 3. How It Explains Bistability

### The $M^n$ Mechanism

When $n > 1$ (e.g., $n = 1.5$):

| Mass (M) | $M^1$ (V1) | $M^{1.5}$ (V2) | Effect |
|----------|------------|----------------|--------|
| 0.5 | 0.5 | 0.35 | Similar |
| 2.0 | 2.0 | 2.83 | V2 heals faster |
| 4.0 | 4.0 | 8.0 | V2 heals much faster |

**Physical Interpretation**: Thick layers are mechanically unstable and break off more easily.

### Scenario 1: Low Temperature (165°C)

- **Physics**: Aluminum is harder ($\Theta(T)$ low) and less sticky ($\Phi(T)$ high)
- **Behavior**: Growth term is tiny, healing dominates
- **$M^n$ effect**: Weak (M stays small)
- **Result**: System trapped in **"Clean Well"** — stable

### Scenario 2: High Temperature (170°C)

- **Physics**: Aluminum softens and becomes sticky
- **Behavior**: System transitions to galled state
- **$M^n$ effect**: Strong — creates oscillations around equilibrium
- **Result**: System in **"Galled Well"** with limit cycle oscillations

### Scenario 3: Transitional Temperature (167.5°C)

- **Physics**: Growth and healing nearly balanced
- **Competition intensity**: Maximum ($I(\beta)$ peaks)
- **Result**: Chaotic behavior — amplified dynamics at transition

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
| $I(\beta)$ | Competition Intensity | Dynamics amplification (V2) |

### Learnable Physics Parameters

| Symbol | Code Name | Initial | Meaning |
|--------|-----------|---------|---------|
| $k_{adh}$ | `k_adh` | 0.05 | Base adhesion rate |
| $\alpha$ | `alpha_tau` | 3.5 | Shear sensitivity (feedback strength) |
| $k_{spall}$ | `k_spall` | 0.8 | Base spalling/healing rate |
| $c_2$ | `E_spall` | 0.5 | Thermal activation energy |
| $n$ | `healing_exponent` | 1.5 | Non-linear healing power (V2) |
| $\kappa$ | `competition_kappa` | 2.0 | Competition intensity at transition (V2) |
| $M_{sat}$ | `M_sat` | 5.0 | Saturation mass threshold |
| $\rho_{crit}$ | `rho_crit` | 0.4 | Critical density for regime shift |
| $k$ | `beta_sharpness` | 15.0 | Transition sharpness |
| — | `noise_lvl` | 0.15 | Process noise amplitude |
| $p_0$ | `prob_detach` | 0.05 | Base detachment probability |
| $\sigma_{output}$ | `output_noise` | 0.05 | Constant output noise (V2) |

### Fixed Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| $T_{ref}$ | 165°C | Reference temperature |
| $\mu_{clean}$ | 0.15 | Clean surface friction |
| $\mu_{galled}$ | 1.0 | Galled surface friction |
| $c_1$ | 0.2 | Thermal softening coefficient |

---

## 5. Implementation Notes

### Code Structure

Model file: `src/models/interactive_galling_model_v2.py`

1. **`compute_rates(T, μ, M, β)`**: Calculates growth, removal rates, and competition intensity
2. **`update_state(M, ρ, T, μ)`**: Evolves mass with $M^n$ healing and competition amplification
3. **`compute_friction(ρ)`**: Computes friction with constant output noise
4. **`simulate_multiple_cycles(T, n_cycles)`**: Runs simulation, tracks competition intensity

### Training

```bash
python scripts/train_interactive_galling_v2.py                    # Train from scratch
python scripts/train_interactive_galling_v2.py --resume-from-best # Fine-tune
python scripts/train_interactive_galling_v2.py --validate-only    # Validation only
```

### Key V2 Innovations

1. **Non-linear Healing ($M^n$)**: Thick layers are mechanically unstable
2. **Competition Intensity**: Amplified dynamics at transition ($\beta \approx 0.5$)
3. **Constant Output Noise**: Oscillations from physics, not noise tuning

### Training Results

**Best Log-Likelihood**: -548.25 at epoch 76/106

| Temperature | Log-Likelihood | Mean β | High Cycles |
|-------------|----------------|--------|-------------|
| 165°C | -125.34 | 0.075 | 0 |
| 167.5°C | -292.25 | 0.412 | 39 |
| 170°C | -130.66 | 0.867 | 129 |

**Key Learned Parameters:**

| Parameter | Learned Value | Physical Meaning |
|-----------|---------------|------------------|
| `healing_exponent` | 1.4485 | Non-linear healing power |
| `competition_kappa` | 1.7516 | Competition intensity factor |
| `output_noise` | 0.1470 | Constant output noise |
| `rho_crit` | 0.2328 | Critical density threshold |
| `k_adh` | 0.2225 | Adhesion rate |
| `k_spall` | 0.5707 | Spalling rate |

**Observations:**
- V2 performs worse than V1 (LL: -548.25 vs -534.10)
- Learned healing exponent (1.45) confirms non-linear removal is helpful
- 167.5°C shows more high cycles than V1 (39 vs 7), indicating better transition capture

### Limitations

- V2 performs worse than V1 (LL: -548 vs -534)
- Constant noise means LOW state is not as stable as training data shows (σ ≈ 0.03)
- $M^n$ creates oscillations but amplitude may not match observations
- Competition intensity is symmetric, but growth/removal may respond differently

---

# Physics-Based Interactive Galling Model V0

## Abstract

**V0** is the baseline physics-based galling model implementing a **closed-loop feedback mechanism** where friction causes galling, and galling causes friction. The model uses:

- **Mass evolution**: Competition between adhesive growth and spalling
- **Shear-driven feedback**: Growth rate scales with $\mu^\alpha$ ($\alpha > 1$)
- **Temperature-dependent rates**: Softening and stickiness functions
- **Constant process noise**: Single noise parameter for mass evolution
- **Massive detachment events**: Probabilistic recovery mechanism

This is the foundational model upon which subsequent versions build.

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

- $M_{sat}$: Saturation mass — When $M \ll M_{sat}$, coverage grows linearly; when $M \gg M_{sat}$, coverage caps at 1.0

### Equation C: Friction Output (Density → Friction)

The friction coefficient is determined by a mixing rule:

$$
\mu_{next} = (1 - \beta_n) \cdot \mu_{clean} + \beta_n \cdot \mu_{galled} + \mathcal{N}(0, \sigma_{output}^2)
$$

Where $\beta_n$ is the **Regime Parameter** (sigmoid transition):

$$
\beta_n = \frac{1}{1 + e^{-k(\rho_n - \rho_{crit})}}
$$

- $\mu_{clean} \approx 0.15$: Steel-on-aluminum friction
- $\mu_{galled} \approx 1.0$: Aluminum-on-aluminum friction
- $\rho_{crit}$: Critical density threshold for regime shift

### Equation D: Massive Detachment Events

Probabilistic event where large chunks break off, acting as a restoring force:

$$
P_{detach} = p_0 \cdot \exp\left( -0.5 \cdot (T - T_{ref}) \right)
$$

If triggered: $M_{next} = 0.5 \cdot M_{next}$

---

## 3. How It Explains Bistability

### Scenario 1: Low Temperature (165°C)

- **Physics**: Aluminum is harder ($\Theta(T)$ low) and less sticky ($\Phi(T)$ high)
- **Behavior**: Growth term is tiny ($(0.15)^{3.5} \approx 0.001$), healing dominates
- **Result**: System trapped in **"Clean Well"** — self-corrects from random spikes

### Scenario 2: High Temperature (170°C)

- **Physics**: Aluminum softens ($\Theta(T)$ high) and becomes sticky ($\Phi(T)$ low)
- **Behavior**: Random spike triggers runaway feedback — growth overwhelms weakened healing
- **Result**: System accelerates into **"Galled Well"** ($\mu \approx 1.0$) and stays there

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

| Symbol | Code Name | Initial | Meaning |
|--------|-----------|---------|---------|
| $k_{adh}$ | `k_adh` | 0.05 | Base adhesion rate |
| $\alpha$ | `alpha_tau` | 3.5 | Shear sensitivity (feedback strength) |
| $k_{spall}$ | `k_spall` | 0.8 | Base spalling/healing rate |
| $c_2$ | `E_spall` | 0.5 | Thermal activation energy |
| $M_{sat}$ | `M_sat` | 5.0 | Saturation mass threshold |
| $\rho_{crit}$ | `rho_crit` | 0.4 | Critical density for regime shift |
| $k$ | `beta_sharpness` | 15.0 | Transition sharpness |
| $\sigma_{noise}$ | `noise_lvl` | 0.15 | Process noise amplitude |
| $p_0$ | `prob_detach` | 0.05 | Base detachment probability |

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

The model methods:

1. **`compute_rates(T, μ)`**: Calculates growth and removal rates
2. **`update_state(M, ρ, T, μ)`**: Evolves mass and density, handles detachment
3. **`compute_friction(ρ)`**: Computes friction from density with constant noise
4. **`simulate_multiple_cycles(T, n_cycles)`**: Runs full simulation

### Key Features

1. **Shear-Driven Feedback**: Growth rate $\propto \mu^\alpha$ creates positive feedback
2. **Temperature-Dependent Healing**: $\Phi(T)$ suppresses healing at high T
3. **Sharp Regime Transitions**: Sigmoid $\beta$ with high sharpness creates bistability
4. **Stochastic Detachment**: Random mass loss enables recovery at low T

### Limitations

- **Constant noise**: Does not capture different stability levels across regimes
- **Uniform variability**: LOW and HIGH states have similar oscillation patterns
- Training data shows LOW state is much more stable than HIGH state

---

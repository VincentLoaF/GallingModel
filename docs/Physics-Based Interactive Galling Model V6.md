# Physics-Based Interactive Galling Model V6

## Abstract

**V6** replaces the physics-based hidden-state models (V3-V5) with a data-driven **Empirical Gaussian Mixture Transition** model. Instead of modeling internal state variables (mass, density, regime parameter), V6 directly learns the transition distribution $P(\Delta\mu \mid T, \mu)$ from observed cycle-to-cycle COF changes.

### Changes from V5

| Aspect | V5 (Growth + Detachment) | V6 (Empirical Transitions) |
|--------|--------------------------|---------------------------|
| State variables | M (mass), ρ (density), β (regime) | None — only observable μ |
| Transition model | Three physics processes | 2-component Gaussian mixture |
| Variability source | Stochastic growth + detachment events | Mixture sampling (stay vs jump) |
| Training method | Monte Carlo KDE likelihood | Direct MLE on observed transitions |
| Training data used | Full trajectory simulation | Individual (T, μ, Δμ) triples |
| Parameter count | 14 learnable | 17 learnable |
| Gradient quality | Noisy (MC sampling) | Exact (analytical) |

**Motivation**: V5 achieved LL=-1000 (2× worse than V3's -479). Key failures: 167.5°C catastrophic (cold start), LOW state too stable (σ=0.007 vs 0.037), many parameters stuck. Analysis of training data revealed that cycle-to-cycle transitions follow clear state-dependent patterns that can be modeled directly without hidden state.

---

## 1. Core Concept: Empirical Transitions

The model treats COF evolution as a **Markov chain** where the next-cycle COF depends only on the current COF and temperature:

$$
\mu(n+1) = \text{clamp}(\mu(n) + \Delta\mu, 0.1, 1.3)
$$

The change $\Delta\mu$ is drawn from a **2-component Gaussian mixture**:

$$
P(\Delta\mu \mid T, \mu) = \pi(T,\mu) \cdot \mathcal{N}(\Delta\mu; d_1, \sigma_1^2) + (1-\pi) \cdot \mathcal{N}(\Delta\mu; d_2, \sigma_2^2)
$$

- **Component 1 ("Stay")**: Small fluctuations within the current regime. Represents normal cycle-to-cycle variability.
- **Component 2 ("Jump")**: Regime transition events. Represents galling spikes (from LOW) or detachment drops (from HIGH).

All mixture parameters ($\pi, d_1, \sigma_1, d_2, \sigma_2$) are analytic functions of $(T, \mu)$ with learnable coefficients.

### Why This Works

Training data analysis shows:
- **165°C**: 113 transitions. 95% are small fluctuations near LOW (σ≈0.037). 5% are large spikes.
- **167.5°C**: 279 transitions. Two clusters (LOW and HIGH) with jumps between them through the sparse MID zone.
- **170°C**: 141 transitions. Mostly moderate fluctuations at HIGH (σ≈0.100). Few transitions from LOW.

This is naturally captured by a mixture: small noise (component 1) plus rare large jumps (component 2), with state-dependent mixing weight.

---

## 2. Governing Equations

### Equation A: Mixing Weight

$$
\pi(T, \mu) = \sigma\left(a_0 + a_T \cdot \Delta T + a_\mu \cdot \mu + a_{\mu\mu} \cdot \mu^2\right)
$$

where $\Delta T = T - T_{ref}$ and $T_{ref} = 160°C$ (below galling onset).

$\pi$ represents the probability of "staying" in the current regime. Higher $\pi$ means more stable behavior. The $\mu^2$ term allows different stickiness at LOW vs HIGH states.

### Equation B: Stay Component (Small Fluctuations)

**Drift:**
$$
d_1(T, \mu) = c_0 + c_\mu \cdot \mu + c_T \cdot \Delta T
$$

**Noise:**
$$
\sigma_1(T, \mu) = \text{softplus}(s_0 + s_\mu \cdot \mu + s_T \cdot \Delta T)
$$

When $c_\mu < 0$, the drift is mean-reverting within each regime. $\sigma_1$ is expected to be small at LOW and larger at HIGH (controlled by $s_\mu > 0$).

### Equation C: Jump Component (Regime Transitions)

**Drift:**
$$
d_2(T, \mu) = j_0 + j_\mu \cdot \mu + j_T \cdot \Delta T
$$

**Noise:**
$$
\sigma_2(T, \mu) = \text{softplus}(v_0 + v_\mu \cdot \mu)
$$

When $j_\mu < 0$: at LOW $\mu$, jumps are upward (galling onset); at HIGH $\mu$, jumps are downward (detachment). The crossover point $\mu^* = -j_0/j_\mu$ determines where jump direction reverses.

### Equation D: Initial Condition

$$
\mu_0(T) = \text{clamp}(m_0 + m_1 \cdot \Delta T, 0.1, 1.3)
$$

Learnable starting COF that varies with temperature.

### Equation E: Full Update

$$
\mu(n+1) = \text{clamp}\left(\mu(n) + \Delta\mu, 0.1, 1.3\right)
$$

where $\Delta\mu$ is sampled from the mixture distribution.

---

## 3. How It Explains Bistability

### Why LOW State is Stable (165°C)

- $\Delta T = 5$, $\mu \approx 0.15$
- High $\pi$ (≈0.95): almost always "stay"
- Component 1: tiny drift (≈0), small $\sigma_1$ (≈0.03)
- Component 2 (rare): positive $d_2$ (spike up), but low probability
- Result: Stable LOW with occasional spikes that return via mean-reversion

### Why HIGH State Oscillates (170°C)

- $\Delta T = 10$, $\mu \approx 0.9$
- Moderate $\pi$ (≈0.85): mostly "stay" but some jumps
- Component 1: moderate $\sigma_1$ (≈0.10) at HIGH state
- Component 2: negative $d_2$ at high $\mu$ (drops down)
- Result: Oscillating HIGH with occasional drops

### Why TRANSITION is Chaotic (167.5°C)

- $\Delta T = 7.5$: intermediate behavior
- From LOW: occasional jumps up (component 2, positive $d_2$)
- From HIGH: occasional drops down (component 2, negative $d_2$)
- MID zone is unstable: both components push toward extremes
- Result: Bistable behavior with unpredictable transitions

### Temperature Effects

| Temperature | $\Delta T$ | π (stay) | Component 1 | Component 2 | Result |
|------------|-----------|----------|-------------|-------------|--------|
| 165°C | 5 | High (~0.95) | Small noise | Rare spike up | Stable LOW |
| 167.5°C | 7.5 | Medium (~0.80) | Moderate noise | Frequent both ways | Bistable |
| 170°C | 10 | High at HIGH (~0.90) | Moderate noise | Rare drops | Oscillating HIGH |

---

## 4. Summary of Variables

### Learnable Parameters (17 total)

| Group | Symbol | Code Name | Initial | Meaning |
|-------|--------|-----------|---------|---------|
| Mixing | $a_0$ | `a0` | 2.0 | Base stay probability (logit) |
| Mixing | $a_T$ | `a_T` | -0.1 | Temperature effect on mixing |
| Mixing | $a_\mu$ | `a_mu` | -1.0 | COF effect on mixing |
| Mixing | $a_{\mu\mu}$ | `a_mu2` | 0.5 | Quadratic COF effect |
| Stay drift | $c_0$ | `c0` | 0.01 | Baseline drift |
| Stay drift | $c_\mu$ | `c_mu` | -0.02 | Mean-reversion strength |
| Stay drift | $c_T$ | `c_T` | 0.001 | Temperature drift effect |
| Stay noise | $s_0$ | `s0` | -3.0 | Base noise level |
| Stay noise | $s_\mu$ | `s_mu` | 1.0 | COF-dependent noise |
| Stay noise | $s_T$ | `s_T` | 0.1 | Temperature noise effect |
| Jump drift | $j_0$ | `j0` | 0.5 | Base jump direction |
| Jump drift | $j_\mu$ | `j_mu` | -1.0 | COF-dependent jump reversal |
| Jump drift | $j_T$ | `j_T` | 0.02 | Temperature jump effect |
| Jump noise | $v_0$ | `v0` | -1.5 | Base jump variability |
| Jump noise | $v_\mu$ | `v_mu` | 0.5 | COF-dependent jump variability |
| Initial | $m_0$ | `mu0_base` | 0.12 | Starting COF at T_ref |
| Initial | $m_1$ | `mu0_T` | 0.005 | Temperature effect on start |

### Fixed Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| $T_{ref}$ | 160°C | Reference temperature (below galling onset) |
| $\mu_{min}$ | 0.1 | Minimum COF clamp |
| $\mu_{max}$ | 1.3 | Maximum COF clamp |

---

## 5. Training

### Method: Direct Maximum Likelihood Estimation

From training data, extract all transition triples $(T_i, \mu_i, \Delta\mu_i)$ where $\Delta\mu_i = \mu(n+1) - \mu(n)$.

**Loss function:**
$$
\mathcal{L} = -\sum_i \log\left[\pi_i \cdot \mathcal{N}(\Delta\mu_i; d_{1,i}, \sigma_{1,i}^2) + (1-\pi_i) \cdot \mathcal{N}(\Delta\mu_i; d_{2,i}, \sigma_{2,i}^2)\right]
$$

where $\pi_i, d_{1,i}, \sigma_{1,i}, d_{2,i}, \sigma_{2,i}$ are evaluated at $(T_i, \mu_i)$.

### Advantages Over V3-V5 Training

| Aspect | V3-V5 | V6 |
|--------|-------|-----|
| Training signal | Simulated trajectory vs observed KDE | Direct transition probability |
| Gradient quality | Noisy (MC sampling through stochastic sim) | Exact (analytical) |
| Computation | 30 simulations × 150 cycles per eval | Single forward pass on 536 samples |
| Bandwidth tuning | Required (KDE bandwidth sensitive) | Not needed |
| Data efficiency | Uses distribution shape only | Uses every individual transition |

### Training Configuration

```bash
python scripts/train_interactive_galling_v6.py                    # Train from scratch
python scripts/train_interactive_galling_v6.py --resume-from-best # Fine-tune
python scripts/train_interactive_galling_v6.py --validate-only    # Validation only
```

### Training Data

~536 transitions total:
- 165°C: 113 transitions
- 167.5°C: 279 transitions
- 170°C: 141 transitions

---

## 6. Implementation Notes

### Code Structure

Model file: `src/models/interactive_galling_model_v6.py`

1. **`transition_params(T, mu)`**: Compute all five mixture parameters
2. **`log_prob(T, mu, delta_mu)`**: Log-likelihood of observed transitions
3. **`sample_step(T, mu)`**: Sample one transition step
4. **`simulate(T, n_cycles)`**: Generate full trajectory
5. **`get_initial_mu(T)`**: Learnable initial condition
6. **`get_physics_params()`**: Returns dict of learned parameter values

### Training Results

**Training:** 5000 epochs, best model at epoch 4999 (converged at final epoch).

| Metric | Value |
|--------|-------|
| Best LL | +651.09 (epoch 4999) |
| Initial LL | +254.14 |
| Training data | 536 transitions (165°C, 167.5°C, 170°C) |

> **Note:** V6 uses direct MLE on individual transitions, producing positive log-likelihoods. This is not directly comparable to V3-V5's simulation-based KDE log-likelihoods (which were negative).

**Per-Temperature Performance:**

| Temperature | LL | Transitions | Assessment |
|-------------|-----|------------|------------|
| 165°C | +199.61 | 113 | Good LOW state capture |
| 167.5°C | +338.94 | 279 | Bistable but transitions too readily to HIGH |
| 170°C | +112.54 | 141 | Good HIGH state oscillations |

**Learned Parameters:**

| Group | Parameter | Learned | Interpretation |
|-------|-----------|---------|----------------|
| Mixing | `a0` | 2.63 | High base stay probability |
| Mixing | `a_T` | -0.22 | Higher T → more jumps |
| Mixing | `a_mu` | -3.37 | Higher μ → more jumps |
| Mixing | `a_mu2` | 2.05 | Quadratic: stabilizes at extreme μ |
| Stay drift | `c0` | 0.028 | Small positive baseline drift |
| Stay drift | `c_mu` | -0.047 | Mean-reverting |
| Stay drift | `c_T` | 0.001 | Minimal temperature effect |
| Stay noise | `s0` | -3.34 | Small base noise |
| Stay noise | `s_mu` | 1.55 | Noise increases with μ |
| Stay noise | `s_T` | 0.089 | Noise increases with T |
| Jump drift | `j0` | -0.021 | Near-zero base jump |
| Jump drift | `j_mu` | -0.196 | Jumps become more negative at high μ |
| Jump drift | `j_T` | 0.010 | Temperature effect on jumps |
| Jump noise | `v0` | -1.08 | Moderate jump variability |
| Jump noise | `v_mu` | 0.147 | Jump variability increases with μ |
| Initial | `mu0_base` | 0.165 | Starting COF near μ_clean |
| Initial | `mu0_T` | 0.009 | Slight increase with T |

**Simulation Assessment:**
- **165°C**: Simulations correctly reproduce stable LOW state with occasional spikes. LOW fraction well-matched.
- **167.5°C**: Model shows bistability but transitions to HIGH too readily. Simulated LOW fraction ~28% vs observed ~49%. The model captures the qualitative bistable behavior but underestimates LOW state persistence.
- **170°C**: Best temperature match. HIGH state oscillations and occasional drops are well captured.

**Key Finding:** Jump drift (d₂) is always negative across the μ range — the model can only produce "drops" from the jump component, not "spikes up." Upward transitions rely on the tail of the stay-component noise distribution, which limits the model's ability to capture sharp LOW→HIGH transitions.

---

## 7. Data Analysis Supporting This Design

### Transition Statistics

**165°C:**
- 54.9% increases, 45.1% decreases
- LOW state (μ<0.25): n=74, Δμ mean=+0.003, std=0.037
- Rare spike events (5 cycles above 0.5)

**167.5°C:**
- 50.5% increases, 49.5% decreases
- LOW state: positive drift (+0.028), suggesting tendency to escalate
- HIGH state: negative drift (-0.041), showing mean-reversion
- MID state: volatile (std=0.229), transition zone

**170°C:**
- LOW state: strong positive drift (+0.028)
- MID state: upward bias (+0.192)
- HIGH state: nearly balanced (-0.009), moderate fluctuations (std=0.100)

### Key Observations

1. The system spends most time at LOW or HIGH — MID is a transition zone, not a stable state
2. Transitions through MID are fast (1-2 cycles), not gradual ramps
3. LOW state variability (σ≈0.037) is real and must be captured
4. HIGH state oscillations (σ≈0.10) come from moderate symmetric noise, not rare detachments

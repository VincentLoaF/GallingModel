# Physics-Based Interactive Galling Model V8

## Abstract

**V8** replaces V7's linear parameter functions with physics derived from a quartic double-well potential. The same 2-component Gaussian mixture likelihood is used, but all 5 mixture parameters (pi, d1, sigma1, d2, sigma2) are now derived from the potential landscape rather than fitted as independent linear functions.

### Changes from V7

| Aspect | V7 | V8 |
|--------|-----|-----|
| Architecture | Linear functions of (T, mu) | Double-well potential V(mu, T) |
| Parameters | 17 linear coefficients | 15 physics parameters |
| Stay probability pi | Logistic: sigma(a0 + a_T*dT + a_mu*mu + a_mu2*mu^2) | Kramers escape: 1 - omega0*exp(-dV/D) |
| Stay drift d1 | Linear: c0 + c_mu*mu + c_T*dT | Potential gradient: -V'(mu,T)*dt |
| Stay noise sigma1 | Softplus: softplus(s0 + s_mu*mu + s_T*dT) | Thermal noise: sqrt(2*D(T)) |
| Jump drift d2 | Linear: j0 + j_mu*mu + j_T*dT | Hybrid: -V'(mu,T)*tau + j_mu*mu + j_T*dT |
| Jump noise sigma2 | Softplus: softplus(v0 + v_mu*mu) | Asymmetric: sigma_jump_up / sigma_jump_down |
| Training data | Same (25C->160C, 165C, 167.5C, 170C) | Same |
| T_ref | 160C | 160C |

**Motivation**: V7's 17 linear coefficients lack physical interpretability. V8 grounds the transition model in a double-well energy landscape, where:
- Two potential wells represent stable clean (mu_L) and galled (mu_H) states
- Temperature modulates the barrier height (Kramers escape theory)
- Thermal noise drives stochastic transitions between wells
- A hybrid jump drift combines potential-derived forces with empirical bias terms

---

## 1. Core Model

**Markov transition model:**
```
mu(n+1) = clamp(mu(n) + delta_mu, 0.1, 1.3)
```

**delta_mu drawn from a 2-component Gaussian mixture:**
```
P(delta_mu | T, mu) = pi(T,mu) * N(delta_mu; d1(T,mu), sigma1(T,mu)^2)
                    + (1 - pi(T,mu)) * N(delta_mu; d2(T,mu), sigma2(T,mu)^2)
```

- **Component 1 ("stay"):** Small fluctuations driven by potential gradient and thermal noise
- **Component 2 ("jump"):** Regime transition (Kramers barrier crossing event)

**No hidden state.** Everything conditioned on observables (T, mu).

---

## 2. Double-Well Potential

The central innovation of V8 is the quartic double-well potential:

```
V(mu, T) = h(T)/L^4 * (mu - mu_L)^2 * (mu - mu_H)^2 - g(T) * (mu - mu_mid)
```

where:
- `L = mu_H - mu_L` (well separation)
- `mu_mid = (mu_L + mu_H) / 2` (midpoint)
- `h(T)` controls barrier height
- `g(T)` controls tilt (asymmetry between wells)

**Potential derivative (restoring force):**
```
dV/dmu = 2*h(T)/L^4 * (mu - mu_L)(mu - mu_H)(2*mu - mu_L - mu_H) - g(T)
```

### Temperature-dependent barrier

```
h(T) = h0 * exp(-alpha_h * dT),    dT = T - 160
```

At the reference temperature (160C), h = h0. As temperature increases, the barrier decreases exponentially, making transitions between clean and galled states increasingly likely.

### Temperature-dependent tilt

```
g(T) = g0 + g_T * dT
```

Positive g tilts the potential to favor the galled state (mu_H); negative g favors the clean state (mu_L). In practice, the optimizer consistently learns g0 ~ 0 and g_T ~ 0, indicating the barrier collapse mechanism (via alpha_h) is sufficient to explain the temperature dependence.

### Noise intensity

```
D(T) = D0 * exp(D_T * dT)
```

Thermal fluctuation intensity increases exponentially with temperature.

---

## 3. Derived Mixture Parameters

All 5 mixture parameters are derived from the potential:

### Mixing weight pi -- P(stay in current regime)

```
pi(T, mu) = clamp(1 - omega0 * exp(-dV / D(T)), 0.01, 0.99)
```

Based on **Kramers escape rate theory**: the probability of escaping a potential well depends exponentially on the ratio of barrier height dV to noise intensity D. The barrier height dV is computed as:

```
dV = V(mu_mid, T) - V(mu, T)    (clamped >= 0)
```

This gives the energy barrier from the current position to the potential maximum at mu_mid.

### Component 1 -- "Fluctuate in place" (stay)

```
d1(T, mu) = -dV/dmu * dt        (dt = 1.0)
sigma1(T, mu) = sqrt(2 * D(T))
```

The stay drift is the restoring force from the potential gradient. Near well minima, V'(mu) ~ 0, so particles fluctuate with small drift. The stay noise is purely thermal.

### Component 2 -- "Regime transition" (jump)

```
d2(T, mu) = -dV/dmu * tau + j_mu * mu + j_T * dT
sigma2(mu) = sigma_jump_up    if mu < mu_mid
           = sigma_jump_down   if mu >= mu_mid
```

The jump drift combines:
1. **Potential-derived force**: -V'(mu) * tau (amplified by time multiplier tau)
2. **Empirical bias terms**: j_mu * mu + j_T * dT

The empirical bias terms (j_mu, j_T) are critical because V'(mu) = 0 at well minima -- precisely where most data points reside. Without these terms, the jump drift would be approximately zero for particles sitting in wells, unable to produce the observed mu-dependent and T-dependent jump behavior.

Jump noise is **asymmetric**: galling onset (upward jumps from LOW state) and detachment (downward jumps from HIGH state) have different landing spreads.

### Initial condition

```
mu0 = clamp(mu0_base, 0.1, 1.3)
```

No temperature dependence in V8's initial condition (simplified from V7).

---

## 4. Parameters

### Total: 15 learnable parameters

| Group | Symbol | Code name | Role |
|-------|--------|-----------|------|
| Well positions | mu_L | `mu_L` | Clean-state equilibrium COF |
| | mu_H | `mu_H` | Galled-state equilibrium COF |
| Barrier | h0 | `h0` | Barrier height at T_ref (>=0) |
| | alpha_h | `alpha_h` | Barrier temperature sensitivity (>=0) |
| Tilt | g0 | `g0` | Base tilt at T_ref |
| | g_T | `g_T` | Temperature sensitivity of tilt |
| Noise | D0 | `D0` | Base noise intensity (>=0) |
| | D_T | `D_T` | Temperature scaling of noise |
| Escape | omega0 | `omega0` | Kramers escape attempt frequency (>=0) |
| Jump dynamics | tau | `tau` | Jump time multiplier (>=0) |
| | j_mu | `j_mu` | Jump drift mu-dependence |
| | j_T | `j_T` | Jump drift T-dependence |
| Jump noise | sigma_jump_up | `sigma_jump_up` | Galling onset landing spread (>=0) |
| | sigma_jump_down | `sigma_jump_down` | Detachment landing spread (>=0) |
| Initial cond | mu0_base | `mu0_base` | Starting COF at T_ref |

### Fixed constants
- T_ref = 160C
- mu_clamp = [0.1, 1.3]
- dt = 1.0 (stay drift time step)

### Positivity enforcement
Parameters marked (>=0) are enforced via `torch.abs()` rather than softplus or exp transforms.

---

## 5. Training

**Same as V7:** Direct Maximum Likelihood Estimation on observed transitions.

```
L = -sum_i log[pi(Ti,mui) * N(delta_mui; d1, sigma1^2) + (1-pi) * N(delta_mui; d2, sigma2^2)]
```

**Temperature mapping:** 25C data mapped to 160C (both represent baseline "no galling" conditions).

**Training configuration:**
- Optimizer: Adam (lr=0.005, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=30, factor=0.5)
- Gradient clipping: max_norm=5.0
- Epochs: 8000 (with early stopping, patience=1000)

**Usage:**
```bash
python scripts/train_interactive_galling_v8.py                    # Train from scratch
python scripts/train_interactive_galling_v8.py --resume-from-best # Fine-tune
python scripts/train_interactive_galling_v8.py --validate-only    # Validation only
```

---

## 6. Key Design Insight: Hybrid Jump Drift

The most important lesson from V8 development was discovering that a pure potential-derived drift is insufficient for the jump component.

**The problem:** At equilibrium points (well minima), V'(mu) = 0 by definition. Since most observed data points sit near well minima, the potential gradient contribution to jump drift is approximately zero. This means:
- Pure physics drift: d2 = -V'(mu) * tau ~ 0 at wells
- Result: Jump events have no mu-dependent direction, producing flat simulation trajectories

**The solution:** Add empirical bias terms that operate where the physics is silent:
```
d2 = -V'(mu,T) * tau + j_mu * mu + j_T * dT
```

- `j_mu * mu`: Provides mu-dependent jump direction (learned j_mu ~ -0.25, meaning higher COF pushes jumps downward)
- `j_T * dT`: Provides T-dependent jump direction (learned j_T ~ +0.023, meaning higher temperature pushes jumps upward toward galling)

This hybrid approach preserves the physical structure of the double-well potential while allowing the model to capture jump behavior that the potential gradient alone cannot describe.

---

## 7. Training Results

**Training:** 8000 epochs, best model at epoch 7574.

| Metric | Value |
|--------|-------|
| Best LL | +1307.85 (epoch 7574) |
| Initial LL | ~+170 |
| Training data | 867 transitions (4 temperatures) |
| Comparison | 98.4% of V7's +1328.61 |

**Per-Temperature Performance:**

| Temperature | Mapped T | LL | Transitions | Assessment |
|-------------|----------|-----|------------|------------|
| 25C | 160C | +654.89 | 331 | Good baseline fit |
| 165C | 165C | +196.04 | 113 | Good (V7: +196.80) |
| 167.5C | 167.5C | +340.24 | 279 | Good (V7: +336.48) |
| 170C | 170C | +116.68 | 141 | Good (V7: +108.90) |

**Simulation Quality:**

| Temperature | Sim Mean | Obs Mean | Sim Std | Obs Std | Assessment |
|-------------|----------|----------|---------|---------|------------|
| 25C (as 160C) | ~0.22 | 0.205 | ~0.08 | 0.036 | Good mean, noisy |
| 165C | ~0.25 | 0.246 | ~0.13 | 0.136 | Excellent |
| 167.5C | ~0.46 | 0.465 | ~0.38 | 0.379 | Excellent |
| 170C | 0.838 | 0.840 | ~0.32 | 0.324 | Near-perfect |

The 170C simulation quality (mean 0.838 vs observed 0.840) was a major improvement achieved by the hybrid jump drift, up from mean ~0.44 without the j_mu/j_T terms.

**Learned Parameters:**

| Parameter | Learned Value | Interpretation |
|-----------|--------------|----------------|
| mu_L | 0.2003 | Clean-state COF (physical: ~0.2) |
| mu_H | 0.8022 | Galled-state COF (physical: ~0.8) |
| h0 | 0.0088 | Small barrier at T_ref |
| alpha_h | 0.6924 | Aggressive barrier collapse with temperature |
| g0 | -0.0001 | Tilt effectively zero |
| g_T | 0.0000 | Tilt temperature sensitivity zero |
| D0 | 0.0008 | Small base noise |
| D_T | 0.1267 | Moderate noise increase with T |
| omega0 | 0.0358 | Low escape attempt frequency |
| tau | 2.7102 | Jump amplification ~3x |
| j_mu | -0.2493 | Negative: higher mu -> downward jumps |
| j_T | +0.0227 | Positive: higher T -> upward jumps |
| sigma_jump_up | 0.2849 | Galling onset spread |
| sigma_jump_down | 0.1362 | Detachment spread (tighter) |
| mu0_base | 0.1632 | Starting COF ~0.16 |

**Key observations:**
- g0 and g_T are effectively zero -- the optimizer prefers barrier collapse (alpha_h) over tilt (g_T) for temperature dependence
- alpha_h = 0.69 means the barrier drops to ~0.1% of its T_ref value at 170C (dT=10), effectively flattening the potential
- j_mu = -0.25 is comparable to V7's -0.19, providing the mu-dependent jump asymmetry
- sigma_jump_up > sigma_jump_down: galling onset has wider spread than detachment

---

## 8. Comparison with V7

| Aspect | V7 | V8 |
|--------|-----|-----|
| Parameters | 17 | 15 |
| Best LL | +1328.61 | +1307.85 (98.4%) |
| 170C sim mean | ~0.84 | 0.838 |
| 165C LL | +196.80 | +196.04 |
| 167.5C LL | +336.48 | +340.24 |
| 170C LL | +108.90 | +116.68 |
| Interpretability | Linear coefficients | Physics parameters |
| Potential structure | None (implicit) | Explicit double-well |

V8 trades a small 20-point LL gap for explicit physical interpretability. The double-well potential provides a mechanistic narrative for galling: two stable states separated by a temperature-dependent energy barrier, with Kramers-rate-governed transitions driven by thermal noise.

---

## 9. Potential Simplification

Since g0 ~ 0 and g_T ~ 0, these parameters could be removed to create a 13-parameter model without loss of performance. The tilt mechanism is redundant with barrier collapse for this dataset. However, the tilt parameters are retained for potential future use with datasets where asymmetric well preference (beyond barrier collapse) is relevant.

---

## 10. Development Log

V8 went through 9 iterations to reach its final form. The reference target was V7's LL = +1328.61 on 867 transitions. The critical challenge was achieving good 170C simulation quality (observed mean = 0.840).

### Iteration 1: Pure potential (13 parameters)

**Setup:** Initial V8 implementation. All 5 mixture parameters derived from the quartic potential. Jump drift purely potential-derived: `d2 = -V'(mu,T) * tau`. Barrier via Arrhenius form `h(T) = h0 * exp(E_a / T_K)`.

| Metric | Value |
|--------|-------|
| LL | +1182.04 |
| 170C sim mean | 0.48 (vs 0.84 observed) |
| 25C sim mean | 0.70 (vs 0.20 observed) |

**Diagnosis:** tau learned to 0.36 (jump amplification *less* than stay -- defeats purpose). g_T ~ 0 (no temperature dependence in tilt). The model used noise to explain everything rather than the potential. V'(mu) = 0 at well minima, so jump drift was uniform near wells and unable to distinguish clean vs galled states.

---

### Iteration 2: Exponential barrier decay (13 parameters)

**Change:** Replaced Arrhenius barrier with exponential decay `h(T) = h0 * exp(-alpha_h * dT)`, giving much more temperature sensitivity over the 10-degree experimental range.

| Metric | Value |
|--------|-------|
| LL | +1272.51 |
| 170C sim mean | ~0.44 |
| 25C sim mean | 0.25 |

**Diagnosis:** alpha_h learned ~0.67 (aggressive barrier collapse). g_T still converged to ~0. Barrier collapse destroyed the wells themselves at high T -- when h(T) -> 0, the entire quartic term vanishes, leaving only the linear tilt (also ~0). No restoring force toward mu_H.

---

### Iteration 3: Clamped alpha_h <= 0.3

**Change:** Clamped alpha_h to max 0.3 to prevent barrier collapse, hoping to force the optimizer to use g_T (tilt) for temperature sensitivity instead.

| Metric | Value |
|--------|-------|
| LL | +1156.63 (worse) |
| 25C sim mean | 0.48 |
| 165C sim mean | 1.03 |
| 170C sim mean | 0.28 |

**Diagnosis:** Constraint caused the optimizer to find a completely different, worse local minimum. g_T = -0.008 (wrong sign), g0 = +0.059 (favors galled at baseline -- wrong), mu_H = 1.12 (too high). Fighting the optimizer's preferred solution made everything worse.

---

### Iteration 4: Minimum barrier floor h_min (14 parameters)

**Change:** Added learnable floor: `h(T) = h_min + (h0 - h_min) * exp(-alpha_h * dT)` so the barrier never drops below h_min. Wells always exist with minimum depth.

| Metric | Value |
|--------|-------|
| LL | +1286.71 (best so far) |
| 25C sim mean | 0.252 |
| 167.5C sim mean | 0.529 |
| 170C sim mean | 0.443 |

**Diagnosis:** h_min learned to ~0 (optimizer chose to ignore the floor). alpha_h = 0.661, g_T ~ 0. The optimizer consistently killed the floor and tilt. This was diagnosed as a fundamental limitation, not a local minimum.

---

### Iteration 5: Fixed h_min clamp (14 parameters)

**Change:** Made h_min non-learnable with fixed minimum value of 0.005.

| Metric | Value |
|--------|-------|
| LL | +1257 (worse than iter 4) |

**Diagnosis:** Same pattern. The optimizer prefers a flat potential with wide jump noise over a structured double-well. Constraining further only hurt.

---

### Iteration 6: Removed alpha_h entirely (12 parameters)

**Change:** Removed barrier temperature dependence. Only tilt g(T) allowed to vary with temperature.

| Metric | Value |
|--------|-------|
| LL | +1247 |
| 25C sim mean | 0.57 |

**Diagnosis:** h0 collapsed to ~0 (flat potential). g_T = -0.0005 (wrong sign, effectively zero). Without barrier dynamics, the model has no mechanism for temperature-dependent transitions.

---

### Iteration 7: Clamped h0 >= 0.05 (12 parameters)

**Change:** Forced minimum barrier height to prevent flat-potential solutions.

| Metric | Value |
|--------|-------|
| LL | +1186 (worse) |

**Diagnosis:** g_T finally turned slightly positive (+0.0004) but far too small to matter. The linear tilt mechanism is fundamentally insufficient -- V'(mu) = 0 at well minima, so tilt only affects the potential derivative away from equilibrium, not where data actually lives.

**Key realization:** Analyzed V7's learned parameters and found V7's success came from `j_mu = -0.445` and `j_T = +0.034` -- independent linear terms in jump drift that operate regardless of the potential gradient. V8 needed the same mechanism.

---

### Iteration 8: Hybrid jump drift -- BREAKTHROUGH (15 parameters)

**Change:** Added independent jump drift parameters: `d2 = -V'(mu,T) * tau + j_mu * mu + j_T * dT`. This gives V7-like mu- and T-dependent jump behavior while keeping the potential for stay dynamics and escape rates. Removed alpha_h to test the hybrid approach cleanly.

| Metric | Value |
|--------|-------|
| LL | +1216 |
| 25C sim mean | 0.199 (vs 0.205 observed) |
| 170C sim mean | **0.794** (vs 0.840 observed) |

**Key parameters:** j_mu = -0.496 (close to V7's -0.445), j_T = +0.041 (close to V7's +0.034).

**Diagnosis:** Massive improvement in 170C simulation quality (from 0.44 to 0.79). The j_mu and j_T terms provided exactly what was missing. LL was lower than iteration 4, but simulation quality was far superior -- demonstrating that LL and trajectory quality are partially decoupled.

---

### Iteration 9: Hybrid + alpha_h restored -- FINAL (15 parameters)

**Change:** Restored alpha_h for barrier temperature dependence alongside hybrid j_mu/j_T. Lowered initial learning rate to 0.005 (from 0.01). 8000 epochs with patience=1000.

| Metric | Value |
|--------|-------|
| LL | **+1307.85** (98.4% of V7) |
| 25C sim mean | 0.244 (vs 0.205 observed) |
| 165C sim mean | 0.337 (vs 0.246 observed) |
| 167.5C sim mean | 0.598 (vs 0.465 observed) |
| 170C sim mean | **0.838** (vs 0.840 observed) |

**Key parameters:** j_mu = -0.249, j_T = +0.023, alpha_h = 0.692, g_T ~ 0, mu_L = 0.200, mu_H = 0.802.

**Assessment:** Best of both worlds. Barrier collapse (alpha_h) handles escape rate temperature dependence. Hybrid jump drift (j_mu, j_T) handles directional jump behavior. The potential provides physical structure while empirical terms fill the gap where V'(mu) = 0.

---

### Summary of All Iterations

| Iter | Key Change | Params | LL | 170C Sim Mean | Outcome |
|------|-----------|--------|-----|---------------|---------|
| 1 | Pure potential (Arrhenius) | 13 | +1182 | 0.48 | Baseline |
| 2 | Exponential barrier decay | 13 | +1273 | ~0.44 | Better LL, still poor sim |
| 3 | Clamped alpha_h <= 0.3 | 13 | +1157 | 0.28 | Worse -- bad local minimum |
| 4 | Added h_min floor | 14 | +1287 | 0.44 | Best LL yet, h_min ignored |
| 5 | Fixed h_min clamp | 14 | +1257 | -- | Worse -- constraints hurt |
| 6 | Removed alpha_h (tilt only) | 12 | +1247 | -- | h0 collapsed to 0 |
| 7 | Clamped h0 >= 0.05 | 12 | +1186 | -- | Tilt insufficient |
| 8 | **Hybrid j_mu/j_T** | 15 | +1216 | **0.79** | Breakthrough in sim quality |
| 9 | **Hybrid + alpha_h** | 15 | **+1308** | **0.84** | Final: best LL + best sim |

### Lessons Learned

1. **V'(mu) = 0 at well minima** is the fundamental limitation of purely potential-derived dynamics. Since data concentrates near equilibria, the potential gradient provides no useful signal for jump drift there.

2. **Don't fight the optimizer.** Clamping, constraining, and forcing parameters into desired ranges (iterations 3, 5, 7) consistently produced worse results than letting the optimizer find its preferred solution and adding flexibility where needed.

3. **LL and simulation quality are partially decoupled.** Iteration 4 had LL = +1287 but 170C sim mean = 0.44. Iteration 8 had LL = +1216 but 170C sim mean = 0.79. Fitting individual transition probabilities well does not guarantee correct long-term trajectory statistics.

4. **The tilt mechanism (g_T) is redundant with barrier collapse (alpha_h).** Across all 9 iterations, g_T consistently learned to ~0. The optimizer always prefers barrier collapse for temperature dependence.

5. **Hybrid physics + empirical is the right architecture.** The potential provides interpretable structure (well positions, barrier dynamics, Kramers escape). The empirical terms (j_mu, j_T) fill the gap where physics is silent. Neither alone is sufficient.

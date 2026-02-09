# Physics-Based Interactive Galling Model V9

## Abstract

**V9** follows standard Kramers-Smoluchowski theory: the double-well potential V(mu) is a **fixed material property** with no temperature dependence. Temperature enters the model ONLY through the thermal noise intensity D(T). This ensures wells always provide strong restoring force, physics (not noise) explains behavior, and the framework is fully consistent with established statistical mechanics.

### Changes from V8

| Aspect | V8 | V9 |
|--------|-----|-----|
| Potential | V(mu,T) = h(T)/L^4*...  (T-dependent) | V(mu) = k/L^4*... (FIXED) |
| Barrier control | h(T) = h0*exp(-alpha_h*dT) collapses at high T | k/16 (constant, material property) |
| Tilt | g(T) = g0 + g_T*dT (always dead) | Removed |
| T enters through | h(T), g(T), D(T) -- redundant/conflicting | D(T) ONLY |
| Parameters | 15 (2 dead) | 12 (all active) |
| Framework | Ad hoc potential modification | Standard Kramers-Smoluchowski |

**Motivation**: V8's barrier collapse via alpha_h=0.69 destroyed the wells at high T, making the model noise-dependent rather than physics-dependent. In standard Kramers theory, the potential is a material property and temperature enters through thermal energy (noise). V9 implements this canonical framework, ensuring wells always confine particles and the restoring force always operates.

---

## 1. Core Model

**Markov transition model:**
```
mu(n+1) = clamp(mu(n) + delta_mu, 0.1, 1.3)
```

**delta_mu drawn from a 2-component Gaussian mixture:**
```
P(delta_mu | T, mu) = pi(T,mu) * N(delta_mu; d1(mu), sigma1(T)^2)
                    + (1 - pi(T,mu)) * N(delta_mu; d2(T,mu), sigma2(mu)^2)
```

- **Component 1 ("stay"):** Potential-confined fluctuations around current well
- **Component 2 ("jump"):** Kramers barrier crossing event

**No hidden state.** Everything conditioned on observables (T, mu).

---

## 2. Fixed Double-Well Potential

The central principle of V9: **the potential is a material property, independent of temperature.**

```
V(mu) = k / L^4 * (mu - mu_L)^2 * (mu - mu_H)^2
```

where:
- `k` = well/barrier stiffness (material property, always > 0)
- `L = mu_H - mu_L` (well separation)
- `mu_L` = clean-state equilibrium COF
- `mu_H` = galled-state equilibrium COF

**Key properties:**
- Well curvature: `V''(mu_L) = V''(mu_H) = 2k/L^2` (always positive, always confining)
- Barrier height: `V(mu_mid) = k/16` (always present)
- Barrier curvature: `V''(mu_mid) = -k/L^2` (negative = unstable, transition zone)

**Potential derivative (restoring force):**
```
dV/dmu = 2k / L^4 * (mu - mu_L)(mu - mu_H)(2*mu - mu_L - mu_H)
```

This restoring force is **always active** -- there is no mechanism for it to collapse or vanish, unlike V8 where alpha_h could destroy it.

---

## 3. Temperature Through Noise Only

Temperature enters the model exclusively through the thermal noise intensity:

```
D(T) = D0 * exp(D_T * dT),    dT = T - 160
```

This is the standard statistical mechanics approach:
- Low T: Small D, particles confined to wells, rare transitions
- High T: Large D, thermal activation enables barrier crossing
- The **ratio dV/D(T)** controls the Kramers escape rate

**Physical justification**: In Kramers-Smoluchowski theory, the escape rate from a potential well is:

```
k_escape ~ omega_0 * exp(-Delta_V / k_B*T)
```

where Delta_V is the fixed barrier height and k_B*T is the thermal energy. In our model, D(T) plays the role of k_B*T. The potential does not change with temperature -- temperature provides thermal energy for barrier crossing.

---

## 4. Derived Mixture Parameters

### Mixing weight pi -- P(stay in current regime)

```
pi(T, mu) = clamp(1 - omega0 * exp(-dV(mu) / D(T)), 0.01, 0.99)
```

where `dV(mu) = V(mu_mid) - V(mu)` is the barrier height from current position.

### Component 1 -- "Fluctuate in place" (stay)

```
d1(mu) = -dV/dmu * dt        (dt = 1.0)
sigma1(T) = sqrt(2 * D(T))
```

The stay drift is the restoring force from the fixed potential. Near wells, this pulls particles back to equilibrium. sigma1 is small thermal noise.

### Component 2 -- "Regime transition" (jump)

```
d2(T, mu) = -dV/dmu * tau + j_mu * mu + j_T * dT
sigma2(mu) = sigma_jump_up    if mu < mu_mid
           = sigma_jump_down   if mu >= mu_mid
```

Hybrid physics + empirical jump drift, same as V8. The j_mu and j_T terms handle directional bias that the potential gradient (zero at wells) cannot provide.

### Initial condition

```
mu0 = clamp(mu0_base, 0.1, 1.3)
```

---

## 5. Parameters

### Total: 12 learnable parameters

| Group | Symbol | Code name | Role |
|-------|--------|-----------|------|
| Well positions | mu_L | `mu_L` | Clean-state equilibrium COF |
| | mu_H | `mu_H` | Galled-state equilibrium COF |
| Material stiffness | k | `k` | Well/barrier stiffness (>=0) |
| Thermal noise | D0 | `D0` | Base noise intensity (>=0) |
| | D_T | `D_T` | Temperature scaling of noise |
| Escape | omega0 | `omega0` | Kramers escape attempt frequency (>=0) |
| Jump dynamics | tau | `tau` | Jump time multiplier (>=0) |
| | j_mu | `j_mu` | Jump drift mu-dependence |
| | j_T | `j_T` | Jump drift T-dependence |
| Jump noise | sigma_jump_up | `sigma_jump_up` | Galling onset spread (>=0) |
| | sigma_jump_down | `sigma_jump_down` | Detachment spread (>=0) |
| Initial cond | mu0_base | `mu0_base` | Starting COF |

### Fixed constants
- T_ref = 160C
- mu_clamp = [0.1, 1.3]
- dt = 1.0

---

## 6. Training

**Same as V8:** Direct Maximum Likelihood Estimation on observed transitions.

```
L = -sum_i log[pi(Ti,mui) * N(delta_mui; d1, sigma1^2) + (1-pi) * N(delta_mui; d2, sigma2^2)]
```

**Training configuration:**
- Optimizer: Adam (lr=0.003, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=30, factor=0.5)
- Gradient clipping: max_norm=5.0
- Epochs: 8000 (with early stopping, patience=1500)

**Usage:**
```bash
python scripts/train_interactive_galling_v9.py                    # Train from scratch
python scripts/train_interactive_galling_v9.py --resume-from-best # Fine-tune
python scripts/train_interactive_galling_v9.py --validate-only    # Validation only
```

---

## 7. Training Results

**Training:** 8000 epochs (early stopped at 5482), best model at epoch 3982.

| Metric | Value |
|--------|-------|
| Best LL | +1219.86 (epoch 3982) |
| Training data | 862 transitions (4 temperatures) |
| Comparison | 93.2% of V8's +1307.85, 91.8% of V7's +1328.61 |

**Per-Temperature Performance:**

| Temperature | Mapped T | LL | Transitions | Assessment |
|-------------|----------|-----|------------|------------|
| 25C | 160C | +688.38 | 331 | Good baseline |
| 165C | 165C | +181.92 | 111 | Decent (V8: +196.04) |
| 167.5C | 167.5C | +256.77 | 279 | Lower (V8: +340.24) |
| 170C | 170C | +92.79 | 141 | Decent (V8: +116.68) |

**Simulation Quality:**

| Temperature | Sim Mean | Obs Mean | Sim Std | Obs Std | Assessment |
|-------------|----------|----------|---------|---------|------------|
| 25C (as 160C) | 0.181 | 0.205 | 0.067 | 0.036 | Good mean |
| 165C | 0.312 | 0.233 | 0.237 | 0.092 | Overshooting |
| 167.5C | 0.615 | 0.465 | 0.333 | 0.379 | Overshooting |
| 170C | 0.843 | 0.840 | 0.261 | 0.324 | Excellent |

**Learned Parameters:**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| mu_L | 0.1585 | Clean-state COF |
| mu_H | 1.0441 | Galled-state COF (wider than V8's 0.802) |
| k | 0.0253 | Small but persistent stiffness |
| D0 | 0.000349 | Small base noise |
| D_T | 0.1184 | Moderate noise increase with T |
| omega0 | 1.2413 | High escape attempt frequency |
| tau | 1.8034 | Jump amplification ~2x |
| j_mu | -0.3888 | Negative: higher mu -> downward jumps |
| j_T | +0.0324 | Positive: higher T -> upward jumps |
| sigma_jump_up | 0.2100 | Galling onset spread |
| sigma_jump_down | 0.2029 | Detachment spread (nearly symmetric) |
| mu0_base | 0.1500 | Starting COF |

**Kramers Analysis:**

| Temperature | D(T) | dV/D | Escape Rate | Regime |
|-------------|------|------|-------------|--------|
| 160C | 0.000349 | 4.53 | 0.013 | Bistable |
| 165C | 0.000631 | 2.51 | 0.101 | Bistable |
| 167.5C | 0.000849 | 1.87 | 0.192 | Active galling |
| 170C | 0.001141 | 1.39 | 0.310 | Active galling |

---

## 8. Comparison with V8

| Aspect | V8 | V9 |
|--------|-----|-----|
| Parameters | 15 (2 dead) | 12 (all active) |
| Best LL | +1307.85 | +1219.86 (93.2%) |
| 170C sim mean | 0.838 | 0.843 |
| 165C sim mean | 0.337 | 0.312 |
| Framework | Ad hoc potential modification | Standard Kramers-Smoluchowski |
| Wells at high T | Collapse (alpha_h=0.69) | Always present (k=0.025) |
| Physics narrative | Potential changes with T | Fixed potential, T enters via noise |
| Journal defensibility | Moderate | Strong (canonical framework) |

**Trade-off:** V9 sacrifices ~88 LL points compared to V8, but gains:
1. Complete physical consistency with Kramers-Smoluchowski theory
2. Wells that always confine particles (no barrier collapse)
3. Fewer parameters (12 vs 15) with no dead parameters
4. A clean narrative: material has fixed energy landscape, temperature provides activation energy

**LL gap analysis:** The gap primarily comes from 167.5C (257 vs 340) and 170C (93 vs 117). V8's ability to collapse the barrier gave it more flexibility at high temperatures. V9's fixed potential is more constrained but more physical.

---

## 9. Key Observations

1. **k is small (0.025):** The material stiffness learned by the optimizer is modest. Barrier height = k/16 = 0.0016. The Kramers mechanism works because D0 is also small (0.00035), giving meaningful dV/D ratios.

2. **mu_H shifted to 1.044:** Wider well separation than V8 (0.802). With a fixed potential, the model needs wider wells to accommodate the range of observed COF values at high temperatures.

3. **omega0 = 1.24:** High escape attempt frequency compensates for the small barrier. In V8, omega0 was 0.036 because alpha_h did most of the work. In V9, omega0 must be large to achieve sufficient transition rates.

4. **sigma_jump_up ~ sigma_jump_down:** Unlike V8 where these were asymmetric (0.285 vs 0.136), V9 learned nearly symmetric jump noise (~0.21 each). The directional asymmetry is handled by j_mu and j_T instead.

5. **165C and 167.5C overshoot:** The model transitions too readily at intermediate temperatures. With a fixed potential, the temperature sensitivity comes entirely from D(T), which may not provide enough dynamic range between 160C and 167.5C. This is the main area for potential improvement.

---

## 10. Development Log

### Iteration 1: Initial V9 implementation (12 parameters)

**Setup:** Fixed potential V(mu) = k/L^4*(mu-mu_L)^2*(mu-mu_H)^2 with no temperature dependence. Temperature enters only through D(T) = D0*exp(D_T*dT). Kramers escape rate pi = 1 - omega0*exp(-dV/D(T)). Hybrid jump drift d2 = -V'(mu)*tau + j_mu*mu + j_T*dT.

**Config:** lr=0.003, 8000 epochs, patience=1500.

| Metric | Value |
|--------|-------|
| LL | +1219.86 (93.2% of V8) |
| 25C sim mean | 0.181 (vs 0.205 observed) |
| 165C sim mean | 0.312 (vs 0.233 observed) |
| 167.5C sim mean | 0.615 (vs 0.465 observed) |
| 170C sim mean | **0.843** (vs 0.840 observed) |

**Diagnosis:** 170C excellent. Intermediate temperatures overshoot -- the fixed potential + D(T) may not provide sufficient dynamic range between baseline and active galling temperatures. The Kramers ratio at 160C (dV/D=4.53) is lower than ideal (>5 preferred for stable baseline).

**Possible improvements for future iterations:**
- Larger initial k to increase dV/D at baseline
- Position-dependent noise sigma1(mu) = sqrt(2*D/V''(mu)) to tighten well oscillations
- Asymmetric well stiffness (different curvature at mu_L vs mu_H)
- Small fixed tilt g*(mu-mu_mid) as a material property (not T-dependent)

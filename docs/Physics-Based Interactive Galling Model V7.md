# Physics-Based Interactive Galling Model V7

## Abstract

**V7** extends V6 by incorporating 25°C (room temperature) data as additional baseline training data. The 25°C data is mapped to T=160°C (ΔT=0) since both represent "no galling" conditions, providing ~331 extra baseline transitions.

### Changes from V6

| Aspect | V6 | V7 |
|--------|-----|-----|
| Training temperatures | 165°C, 167.5°C, 170°C | 25°C→160°C, 165°C, 167.5°C, 170°C |
| Total transitions | ~536 | ~867 (+331 baseline) |
| Architecture | Unchanged | Unchanged |
| Parameters | 17 learnable | 17 learnable |
| T_ref | 160°C | 160°C |

**Motivation**: V6 had limited baseline (ΔT=0) data — only interpolated behavior between 165°C and the assumed 160°C reference. Adding 25°C data (which behaves like baseline since no thermal activation occurs at room temperature) provides direct observations of "no galling" behavior, improving the model's ability to learn the low-state transition statistics.

---

## 1. Core Model (Same as V6)

**Markov transition model:**
```
μ(n+1) = clamp(μ(n) + Δμ, 0.1, 1.3)
```

**Δμ drawn from a 2-component Gaussian mixture:**
```
P(Δμ | T, μ) = π(T,μ) · N(Δμ; d₁(T,μ), σ₁(T,μ)²)
              + (1 - π(T,μ)) · N(Δμ; d₂(T,μ), σ₂(T,μ)²)
```

- **Component 1 ("stay"):** Small fluctuations within current regime
- **Component 2 ("jump"):** Regime transition event

**No hidden state.** Everything conditioned on observables (T, μ).

---

## 2. Temperature Mapping

The key innovation in V7 is treating 25°C data as if it were collected at 160°C:

| Actual Temperature | Model Temperature | ΔT = T - 160 | Behavior |
|-------------------|-------------------|--------------|----------|
| 25°C | 160°C | 0 | Baseline (no galling) |
| 165°C | 165°C | 5 | Mostly stable LOW |
| 167.5°C | 167.5°C | 7.5 | Bistable |
| 170°C | 170°C | 10 | Mostly HIGH |

**Physical justification:** Both 25°C and 160°C are below the galling onset temperature. At 25°C, there is no thermal activation of adhesion, so the material behaves in its "clean" baseline state. Since 160°C is the model's reference point (ΔT=0), mapping 25°C→160°C assigns the correct baseline behavior to the 25°C observations.

---

## 3. Training Data Statistics

**25°C (mapped to 160°C):**
- 332 cycles, 331 transitions
- Mean COF: 0.2046, std: 0.0355
- Range: [0.1294, 0.5978]
- Δμ: mean ≈ 0, std = 0.041
- 94.9% of cycles in LOW state (μ < 0.25)

**165°C:**
- 114 cycles, 113 transitions
- Mean COF: 0.2461, std: 0.1364

**167.5°C:**
- 280 cycles, 279 transitions
- Mean COF: 0.4650, std: 0.3794

**170°C:**
- 142 cycles, 141 transitions
- Mean COF: 0.8396, std: 0.3236

**Total: 867 transitions** (vs 533 in V6)

---

## 4. Parameterization (Same as V6)

### Mixing weight π — P(stay in current regime)
```
π(T,μ) = σ(a₀ + a_T·ΔT + a_μ·μ + a_μμ·μ²)
```

### Component 1 — "Fluctuate in place"
```
d₁(T,μ) = c₀ + c_μ·μ + c_T·ΔT
σ₁(T,μ) = softplus(s₀ + s_μ·μ + s_T·ΔT)
```

### Component 2 — "Regime transition"
```
d₂(T,μ) = j₀ + j_μ·μ + j_T·ΔT
σ₂(T,μ) = softplus(v₀ + v_μ·μ)
```

### Initial condition
```
μ₀(T) = clamp(m₀ + m₁·ΔT, 0.1, 1.3)
```

### Total: 17 learnable parameters

| Group | Params | Code names |
|-------|--------|------------|
| Mixing weight | a₀, a_T, a_μ, a_μμ | `a0, a_T, a_mu, a_mu2` |
| Stay drift | c₀, c_μ, c_T | `c0, c_mu, c_T` |
| Stay noise | s₀, s_μ, s_T | `s0, s_mu, s_T` |
| Jump drift | j₀, j_μ, j_T | `j0, j_mu, j_T` |
| Jump noise | v₀, v_μ | `v0, v_mu` |
| Initial cond | m₀, m₁ | `mu0_base, mu0_T` |

### Fixed constants
- T_ref = 160°C
- μ_clamp = [0.1, 1.3]

---

## 5. Training

**Same as V6:** Direct Maximum Likelihood Estimation on observed transitions.

```
L = -Σᵢ log[π(Tᵢ,μᵢ)·N(Δμᵢ; d₁, σ₁²) + (1-π)·N(Δμᵢ; d₂, σ₂²)]
```

**Usage:**
```bash
python scripts/train_interactive_galling_v7.py                    # Train from scratch
python scripts/train_interactive_galling_v7.py --resume-from-best # Fine-tune
python scripts/train_interactive_galling_v7.py --validate-only    # Validation only
```

---

## 6. Expected Improvements from V6

1. **Better LOW-state variability:** 331 extra baseline transitions should help the model learn the correct noise level for the LOW state (target σ≈0.037).

2. **More stable baseline behavior:** With direct observations at ΔT=0, the model should learn that baseline behavior is very stable with rare spikes.

3. **Improved temperature extrapolation:** Having data at the actual reference point (ΔT=0) improves the model's ability to interpolate and extrapolate across temperatures.

---

## 7. Training Results

**Training:** 5000 epochs, best model at epoch 4798.

| Metric | Value |
|--------|-------|
| Best LL | +1328.61 (epoch 4798) |
| Initial LL | +791.54 |
| Training data | 867 transitions (4 temperatures) |
| Comparison | ~2× V6's +651.09 (due to added baseline data) |

**Per-Temperature Performance:**

| Temperature | Mapped T | LL | Transitions | Assessment |
|-------------|----------|-----|------------|------------|
| 25°C | 160°C | +686.43 | 331 | Poor — too many jump events |
| 165°C | 165°C | +196.80 | 113 | Good (similar to V6's +199.61) |
| 167.5°C | 167.5°C | +336.48 | 279 | Good (similar to V6's +338.94) |
| 170°C | 170°C | +108.90 | 141 | Good (similar to V6's +112.54) |

**25°C Modeling Problem:**

| Metric | Simulated | Observed | Gap |
|--------|-----------|----------|-----|
| Mean COF | ~0.22 | 0.2046 | Slight overestimate |
| Std COF | 0.0965 | 0.0355 | 2.7× too noisy |
| LOW fraction | 74.5% | 94.9% | 20% too many transitions |
| Jump rate | ~3.2%/cycle | ~0.3-0.6%/cycle | 5-10× too many jumps |

**Root cause:** At ΔT=0 (baseline), P(stay) ≈ 0.968 — the model allows ~3.2% jump events per cycle. Observed 25°C data shows only 1-2 spikes in 332 cycles (~0.3-0.6% jump rate). The linear ΔT dependence in the mixing weight π = σ(a₀ + a_T·ΔT + ...) cannot create a sharp enough nonlinearity between the baseline "no galling" regime and the active galling temperatures (165-170°C). Pushing a₀ higher to suppress baseline jumps would also suppress necessary jumps at higher temperatures.

**Learned Parameters (changes vs V6):**

| Parameter | V6 Value | V7 Value | Change |
|-----------|----------|----------|--------|
| `a0` | 2.63 | 2.87 | ↑ Slightly higher stay prob |
| `a_T` | -0.22 | -0.24 | Similar |
| `a_mu` | -3.37 | -3.52 | Similar |
| `s0` | -3.34 | -3.41 | Slightly less noise |
| `j_mu` | -0.196 | -0.189 | Similar |

Parameters for shared temperatures (165, 167.5, 170°C) are nearly identical to V6, confirming the added 25°C data mainly affects the baseline region without disrupting higher-temperature fits.

**Assessment:** V7 successfully preserved V6's performance at 165-170°C while adding baseline data. However, the 25°C modeling is poor because the linear temperature parameterization cannot capture the sharp threshold between "no galling" and "active galling." Potential fixes include nonlinear temperature terms (ΔT²), sigmoid-based threshold in mixing weight, or per-regime offsets.

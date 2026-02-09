"""
Interactive Galling Model V6 - Empirical Gaussian Mixture Transitions

V6 replaces V5's physics-based growth+detachment with a data-driven approach.
Instead of modeling hidden state (mass, density), V6 directly learns the
transition distribution P(Δμ | T, μ) from observed cycle-to-cycle COF changes.

Core idea:
    μ(n+1) = clamp(μ(n) + Δμ, 0.1, 1.3)
    Δμ ~ π·N(d₁, σ₁²) + (1-π)·N(d₂, σ₂²)

Two mixture components:
    1. "Stay" — small fluctuations within current regime
    2. "Jump" — regime transition event (spike up or drop down)

All parameters (π, d₁, σ₁, d₂, σ₂) are analytic functions of (T, μ)
with learnable coefficients. 17 total learnable parameters.

Training: Direct MLE on observed transitions. No MC sampling, no KDE.

Date: 2026-02-02
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class InteractiveGallingModelV6(nn.Module):
    """
    Empirical Gaussian Mixture Transition Model for Galling Prediction.

    Directly models P(Δμ | T, μ) as a 2-component Gaussian mixture
    where all component parameters are smooth functions of temperature
    and current friction coefficient.

    No hidden state (mass, density, beta). All variability comes from
    the stochastic mixture sampling.
    """

    T_REF = 160.0      # Reference temperature (below galling onset)
    MU_MIN = 0.1       # Minimum COF clamp
    MU_MAX = 1.3       # Maximum COF clamp

    def __init__(
        self,
        param_init: Optional[Dict[str, float]] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device

        if param_init is None:
            param_init = {
                # --- Mixing weight: π = σ(a0 + a_T·ΔT + a_mu·μ + a_mu2·μ²) ---
                'a0': 2.0,        # Intercept (positive → high π → mostly "stay")
                'a_T': -0.1,      # Higher T → slightly less sticky
                'a_mu': -1.0,     # Higher μ → less sticky (more transitions at HIGH)
                'a_mu2': 0.5,     # Quadratic: both extremes slightly more sticky

                # --- Stay component drift: d₁ = c0 + c_mu·μ + c_T·ΔT ---
                'c0': 0.01,       # Small positive baseline drift
                'c_mu': -0.02,    # Slight mean-reversion (negative: HIGH drifts down)
                'c_T': 0.001,     # Higher T → slightly more positive drift

                # --- Stay component noise: σ₁ = softplus(s0 + s_mu·μ + s_T·ΔT) ---
                's0': -3.0,       # Base noise (softplus(-3) ≈ 0.05)
                's_mu': 1.0,      # More noise at higher COF
                's_T': 0.1,       # More noise at higher T

                # --- Jump component drift: d₂ = j0 + j_mu·μ + j_T·ΔT ---
                'j0': 0.5,        # Positive baseline (jump up from LOW)
                'j_mu': -1.0,     # Negative slope (at HIGH μ, jump direction reverses)
                'j_T': 0.02,      # Higher T → slightly larger upward jumps

                # --- Jump component noise: σ₂ = softplus(v0 + v_mu·μ) ---
                'v0': -1.5,       # Base jump variability (softplus(-1.5) ≈ 0.20)
                'v_mu': 0.5,      # More variability at higher COF

                # --- Initial condition: μ₀ = clamp(m0 + m1·ΔT, 0.1, 1.3) ---
                'mu0_base': 0.12, # Starting COF at T_ref
                'mu0_T': 0.005,   # Higher T → slightly higher start
            }

        self.params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for name, value in param_init.items()
        })

    def transition_params(self, T: torch.Tensor, mu: torch.Tensor):
        """
        Compute all mixture parameters for given (T, μ).

        Returns:
            pi: mixing weight P(stay) in (0, 1)
            d1, sigma1: stay component drift and noise
            d2, sigma2: jump component drift and noise
        """
        dT = T - self.T_REF

        # Mixing weight
        logit_pi = (self.params['a0']
                    + self.params['a_T'] * dT
                    + self.params['a_mu'] * mu
                    + self.params['a_mu2'] * mu ** 2)
        pi = torch.sigmoid(logit_pi)

        # Stay component
        d1 = (self.params['c0']
              + self.params['c_mu'] * mu
              + self.params['c_T'] * dT)
        sigma1 = torch.nn.functional.softplus(
            self.params['s0']
            + self.params['s_mu'] * mu
            + self.params['s_T'] * dT
        )

        # Jump component
        d2 = (self.params['j0']
              + self.params['j_mu'] * mu
              + self.params['j_T'] * dT)
        sigma2 = torch.nn.functional.softplus(
            self.params['v0']
            + self.params['v_mu'] * mu
        )

        return pi, d1, sigma1, d2, sigma2

    def log_prob(self, T: torch.Tensor, mu: torch.Tensor, delta_mu: torch.Tensor):
        """
        Compute log P(Δμ | T, μ) for observed transitions.

        Args:
            T: temperature (scalar or batch)
            mu: current COF μ(n) (batch)
            delta_mu: observed change Δμ = μ(n+1) - μ(n) (batch)

        Returns:
            log_likelihood: per-sample log-probabilities (batch)
        """
        pi, d1, sigma1, d2, sigma2 = self.transition_params(T, mu)

        # Log-probability under each component
        log_p1 = self._gaussian_log_prob(delta_mu, d1, sigma1)
        log_p2 = self._gaussian_log_prob(delta_mu, d2, sigma2)

        # Log-sum-exp for mixture
        log_pi = torch.log(pi + 1e-8)
        log_one_minus_pi = torch.log(1 - pi + 1e-8)

        log_mix = torch.logsumexp(
            torch.stack([log_pi + log_p1, log_one_minus_pi + log_p2], dim=0),
            dim=0
        )

        return log_mix

    @staticmethod
    def _gaussian_log_prob(x, mean, sigma):
        """Log-probability of x under N(mean, sigma²)."""
        return -0.5 * math.log(2 * math.pi) - torch.log(sigma + 1e-8) \
               - 0.5 * ((x - mean) / (sigma + 1e-8)) ** 2

    def sample_step(self, T: torch.Tensor, mu: torch.Tensor):
        """
        Sample one transition step.

        Args:
            T: temperature (scalar)
            mu: current COF (scalar or batch)

        Returns:
            mu_next: next COF (same shape as mu)
            component: which component was sampled (0=stay, 1=jump)
        """
        pi, d1, sigma1, d2, sigma2 = self.transition_params(T, mu)

        # Sample component
        is_stay = torch.rand_like(mu) < pi
        component = (~is_stay).float()

        # Sample delta from chosen component
        noise = torch.randn_like(mu)
        delta_stay = d1 + sigma1 * noise
        delta_jump = d2 + sigma2 * noise

        delta_mu = torch.where(is_stay, delta_stay, delta_jump)

        mu_next = torch.clamp(mu + delta_mu, self.MU_MIN, self.MU_MAX)

        return mu_next, component

    def get_initial_mu(self, T: float) -> torch.Tensor:
        """Get initial COF for a given temperature."""
        dT = T - self.T_REF
        mu0 = self.params['mu0_base'] + self.params['mu0_T'] * dT
        return torch.clamp(mu0, self.MU_MIN, self.MU_MAX).unsqueeze(0)

    def simulate(
        self,
        T: float,
        n_cycles: int = 150,
        mu_init: Optional[float] = None
    ):
        """
        Simulate a COF trajectory.

        Args:
            T: temperature in °C
            n_cycles: number of cycles to simulate
            mu_init: optional initial COF (uses learned μ₀(T) if None)

        Returns:
            dict with mu_history, component_history, pi_history, etc.
        """
        device = next(self.parameters()).device
        T_tensor = torch.tensor(T, dtype=torch.float32, device=device)

        if mu_init is not None:
            mu = torch.tensor([mu_init], dtype=torch.float32, device=device)
        else:
            mu = self.get_initial_mu(T).to(device)

        mu_list = []
        component_list = []
        pi_list = []
        d1_list = []
        d2_list = []
        sigma1_list = []
        sigma2_list = []

        for _ in range(n_cycles):
            pi, d1, sigma1, d2, sigma2 = self.transition_params(T_tensor, mu)

            # Store pre-step diagnostics
            pi_list.append(pi.detach())
            d1_list.append(d1.detach())
            d2_list.append(d2.detach())
            sigma1_list.append(sigma1.detach())
            sigma2_list.append(sigma2.detach())

            # Take step
            mu, component = self.sample_step(T_tensor, mu)

            mu_list.append(mu.detach())
            component_list.append(component.detach())

        return {
            'mu_history': torch.cat(mu_list),
            'component_history': torch.cat(component_list),
            'pi_history': torch.cat(pi_list),
            'd1_history': torch.cat(d1_list),
            'd2_history': torch.cat(d2_list),
            'sigma1_history': torch.cat(sigma1_list),
            'sigma2_history': torch.cat(sigma2_list),
        }

    def get_physics_params(self):
        """Return all parameters as a dict (for compatibility with trainer logging)."""
        return {name: param.item() for name, param in self.params.items()}


# --- Verification Script ---
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Interactive Galling Model V6 - Empirical Transition Model")
    print("=" * 70)

    model = InteractiveGallingModelV6()

    print("\nLearnable parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:15s} = {value:.4f}")
    print(f"\nTotal: {len(model.params)} parameters")

    # Simulate all temperatures
    temps = [165, 167.5, 170]
    results = {}

    for temp in temps:
        with torch.no_grad():
            res = model.simulate(T=temp, n_cycles=150)
        results[temp] = {k: v.cpu().numpy() for k, v in res.items()}

        mu = results[temp]['mu_history']
        comp = results[temp]['component_history']
        pi = results[temp]['pi_history']

        print(f"\n{temp}°C:")
        print(f"  COF: mean={mu.mean():.4f}, std={mu.std():.4f}")
        print(f"  Range: [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  Jump events: {comp.sum():.0f}/{len(comp)} ({comp.mean()*100:.1f}%)")
        print(f"  Mean P(stay): {pi.mean():.4f}")

    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Interactive Galling Model V6 - Verification', fontsize=14)

    colors = {165: '#003f5c', 167.5: '#bc5090', 170: '#ffa600'}

    for idx, temp in enumerate(temps):
        res = results[temp]
        cycles = np.arange(1, 151)

        # Col 0: COF trajectory
        axes[idx, 0].plot(cycles, res['mu_history'], color=colors[temp], linewidth=1)
        axes[idx, 0].set_title(f'{temp}°C - COF')
        axes[idx, 0].set_ylabel('COF')
        axes[idx, 0].set_ylim(0, 1.4)
        axes[idx, 0].grid(alpha=0.3)

        # Col 1: P(stay) over time
        axes[idx, 1].plot(cycles, res['pi_history'], color=colors[temp], linewidth=1)
        axes[idx, 1].set_title(f'{temp}°C - P(stay)')
        axes[idx, 1].set_ylabel('π')
        axes[idx, 1].set_ylim(0, 1)
        axes[idx, 1].grid(alpha=0.3)

        # Col 2: Component (stay=0, jump=1) events
        jump_cycles = cycles[res['component_history'] > 0.5]
        axes[idx, 2].scatter(jump_cycles, np.ones_like(jump_cycles),
                            c='red', s=15, alpha=0.7, label='Jump')
        stay_cycles = cycles[res['component_history'] < 0.5]
        axes[idx, 2].scatter(stay_cycles, np.zeros_like(stay_cycles),
                            c='green', s=5, alpha=0.3, label='Stay')
        axes[idx, 2].set_title(f'{temp}°C - Events')
        axes[idx, 2].set_ylabel('Component')
        axes[idx, 2].set_ylim(-0.3, 1.3)
        axes[idx, 2].legend(fontsize=8)
        axes[idx, 2].grid(alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('Cycle')

    plt.tight_layout()
    plt.savefig('v6_model_test.png', dpi=150)
    print("\nSaved: v6_model_test.png")

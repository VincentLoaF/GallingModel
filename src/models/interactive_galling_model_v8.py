"""
Interactive Galling Model V8 - Double-Well Potential Galling Model

V8 replaces V7's linear parameter functions with physics derived from a
quartic double-well potential. The same 2-component Gaussian mixture likelihood
is used, but all 5 mixture parameters (pi, d1, sigma1, d2, sigma2) are now
derived from the potential landscape rather than fitted as linear functions.

Double-well potential:
    V(mu, T) = h(T)/L^4 * (mu - mu_L)^2 * (mu - mu_H)^2 - g(T)*(mu - mu_mid)

where L = mu_H - mu_L, mu_mid = (mu_L + mu_H) / 2

Barrier height: h(T) = h0 * exp(-alpha_h * dT)
Tilt: g(T) = g0 + g_T * dT, dT = T - 160
Noise intensity: D(T) = D0 * exp(D_T * dT)

Derived mixture parameters:
    pi (stay prob)   = clamp(1 - omega0*exp(-dV_barrier/D(T)), 0.01, 0.99)
    d1 (stay drift)  = -V'(mu, T) * dt
    sigma1 (stay noise) = sqrt(2*D(T))
    d2 (jump drift)  = -V'(mu, T) * tau + j_mu * mu + j_T * dT
    sigma2 (jump noise) = sigma_jump_up if mu < mu_mid else sigma_jump_down

15 learnable parameters, all with physical interpretation.

Date: 2026-02-09
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple


class InteractiveGallingModelV8(nn.Module):
    """
    Double-Well Potential Galling Model.

    Models P(delta_mu | T, mu) as a 2-component Gaussian mixture where all
    component parameters are derived from a quartic double-well potential
    V(mu, T). The potential has two minima at mu_L (clean) and mu_H (galled),
    with temperature-dependent barrier height and tilt.
    """

    # Fixed constants
    T_REF = 160.0       # Reference temperature (C)
    MU_MIN = 0.1        # Minimum COF clamp
    MU_MAX = 1.3        # Maximum COF clamp
    DT = 1.0            # Time step for stay drift

    def __init__(
        self,
        param_init: Optional[Dict[str, float]] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device

        if param_init is None:
            param_init = {
                'mu_L': 0.18,              # Clean-state equilibrium COF
                'mu_H': 0.90,              # Galled-state equilibrium COF
                'h0': 0.1,                 # Barrier height at T_ref
                'alpha_h': 0.1,            # Barrier temperature sensitivity (1/°C)
                'g0': -0.01,               # Base tilt (favors clean at T_ref)
                'g_T': 0.005,              # Temperature sensitivity of tilt
                'D0': 0.001,               # Base noise intensity
                'D_T': 0.1,                # Temperature scaling of noise
                'omega0': 0.1,             # Escape attempt frequency
                'tau': 3.0,                # Jump time multiplier
                'j_mu': -0.3,              # Jump drift μ-dependence
                'j_T': 0.03,               # Jump drift T-dependence
                'sigma_jump_up': 0.15,     # Galling onset landing spread
                'sigma_jump_down': 0.10,   # Detachment landing spread
                'mu0_base': 0.15,          # Starting COF at T_ref
            }

        self.params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for name, value in param_init.items()
        })

    def _get_safe_params(self):
        """Return positivity-enforced parameters."""
        mu_L = self.params['mu_L']
        mu_H = self.params['mu_H']
        h0 = torch.abs(self.params['h0'])
        alpha_h = torch.abs(self.params['alpha_h'])
        g0 = self.params['g0']
        g_T = self.params['g_T']
        D0 = torch.abs(self.params['D0'])
        D_T = self.params['D_T']
        omega0 = torch.abs(self.params['omega0'])
        tau = torch.abs(self.params['tau'])
        j_mu = self.params['j_mu']
        j_T = self.params['j_T']
        sigma_jump_up = torch.abs(self.params['sigma_jump_up'])
        sigma_jump_down = torch.abs(self.params['sigma_jump_down'])
        return mu_L, mu_H, h0, alpha_h, g0, g_T, D0, D_T, omega0, tau, j_mu, j_T, sigma_jump_up, sigma_jump_down

    def _barrier_and_tilt(self, T: torch.Tensor):
        """Compute h(T) and g(T).

        h(T) = h0 * exp(-alpha_h * dT): barrier decreases with temperature.
        g(T) = g0 + g_T * dT: tilt increases with temperature.
        """
        _, _, h0, alpha_h, g0, g_T, _, _, _, _, _, _, _, _ = self._get_safe_params()
        dT = T - self.T_REF
        exp_arg = torch.clamp(-alpha_h * dT, -20.0, 20.0)
        h = h0 * torch.exp(exp_arg)
        g = g0 + g_T * dT
        return h, g

    def noise_intensity(self, T: torch.Tensor) -> torch.Tensor:
        """Compute noise intensity D(T) = D0 * exp(D_T * dT)."""
        _, _, _, _, _, _, D0, D_T, _, _, _, _, _, _ = self._get_safe_params()
        dT = T - self.T_REF
        exp_arg = torch.clamp(D_T * dT, -20.0, 20.0)
        return D0 * torch.exp(exp_arg)

    def potential(self, mu: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute the double-well potential V(mu, T).

        V(mu, T) = h(T)/L^4 * (mu - mu_L)^2 * (mu - mu_H)^2 - g(T)*(mu - mu_mid)
        """
        mu_L, mu_H, _, _, _, _, _, _, _, _, _, _, _, _ = self._get_safe_params()
        h, g = self._barrier_and_tilt(T)

        L = mu_H - mu_L
        L4 = L ** 4 + 1e-8
        mu_mid = (mu_L + mu_H) / 2.0

        V = h / L4 * (mu - mu_L) ** 2 * (mu - mu_H) ** 2 - g * (mu - mu_mid)
        return V

    def potential_derivative(self, mu: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Analytical derivative dV/dmu.

        dV/dmu = 2h/L^4 * (mu - mu_L)(mu - mu_H)(2*mu - mu_L - mu_H) - g
        """
        mu_L, mu_H, _, _, _, _, _, _, _, _, _, _, _, _ = self._get_safe_params()
        h, g = self._barrier_and_tilt(T)

        L = mu_H - mu_L
        L4 = L ** 4 + 1e-8

        dVdmu = 2.0 * h / L4 * (mu - mu_L) * (mu - mu_H) * (2.0 * mu - mu_L - mu_H) - g
        return dVdmu

    def barrier_height(self, mu: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute effective barrier height dV = V(mu_mid, T) - V(mu, T), clamped >= 0.
        """
        mu_L = self.params['mu_L']
        mu_H = self.params['mu_H']
        mu_mid = (mu_L + mu_H) / 2.0

        V_mid = self.potential(mu_mid.unsqueeze(0) if mu_mid.dim() == 0 else mu_mid, T)
        V_mu = self.potential(mu, T)

        # Expand V_mid to match shape of V_mu if needed
        if V_mid.dim() == 0 or (V_mid.shape != V_mu.shape):
            V_mid = V_mid.expand_as(V_mu)

        dV = V_mid - V_mu
        dV = torch.clamp(dV, min=0.0)
        return dV

    def transition_params(self, T: torch.Tensor, mu: torch.Tensor):
        """
        Compute all mixture parameters derived from the double-well potential.

        Returns:
            pi: mixing weight P(stay) in (0.01, 0.99)
            d1: stay component drift
            sigma1: stay component noise
            d2: jump component drift
            sigma2: jump component noise
        """
        mu_L, mu_H, _, _, _, _, _, _, omega0, tau, j_mu, j_T, sigma_jump_up, sigma_jump_down = self._get_safe_params()
        mu_mid = (mu_L + mu_H) / 2.0

        # Noise intensity
        D = self.noise_intensity(T)

        # Barrier height for escape probability
        dV = self.barrier_height(mu, T)

        # pi (stay probability) = clamp(1 - omega0 * exp(-dV / D), 0.01, 0.99)
        exp_arg = torch.clamp(-dV / (D + 1e-8), -20.0, 20.0)
        pi = torch.clamp(1.0 - omega0 * torch.exp(exp_arg), 0.01, 0.99)

        # Stay drift: d1 = -V'(mu, T) * dt
        dVdmu = self.potential_derivative(mu, T)
        d1 = -dVdmu * self.DT

        # Stay noise: sigma1 = sqrt(2 * D(T))
        sigma1 = torch.sqrt(2.0 * D + 1e-8).expand_as(mu)

        # Jump drift: d2 = -V'(mu, T) * tau + j_mu * mu + j_T * dT
        # The j_mu and j_T terms add μ- and T-dependent bias to jump events,
        # capturing that anomalous cycles have asymmetric behavior not fully
        # described by the potential gradient alone (which is zero at well minima).
        dT = T - self.T_REF
        d2 = -dVdmu * tau + j_mu * mu + j_T * dT

        # Jump noise: sigma2 depends on which side of mu_mid
        sigma2 = torch.where(mu < mu_mid, sigma_jump_up, sigma_jump_down)

        return pi, d1, sigma1, d2, sigma2

    def log_prob(self, T: torch.Tensor, mu: torch.Tensor, delta_mu: torch.Tensor):
        """
        Compute log P(delta_mu | T, mu) for observed transitions.

        Args:
            T: temperature (scalar or batch)
            mu: current COF mu(n) (batch)
            delta_mu: observed change delta_mu = mu(n+1) - mu(n) (batch)

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
        """Log-probability of x under N(mean, sigma^2)."""
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
        """Get initial COF for a given temperature (no T dependence in V8)."""
        mu0 = self.params['mu0_base']
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
            T: temperature in C
            n_cycles: number of cycles to simulate
            mu_init: optional initial COF (uses learned mu0 if None)

        Returns:
            dict with mu_history, component_history, pi_history,
            d1_history, d2_history, sigma1_history, sigma2_history,
            force_history
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
        force_list = []

        for _ in range(n_cycles):
            pi, d1, sigma1, d2, sigma2 = self.transition_params(T_tensor, mu)

            # Force = -dV/dmu (restoring force from potential)
            force = -self.potential_derivative(mu, T_tensor)

            # Store pre-step diagnostics
            pi_list.append(pi.detach())
            d1_list.append(d1.detach())
            d2_list.append(d2.detach())
            sigma1_list.append(sigma1.detach())
            sigma2_list.append(sigma2.detach())
            force_list.append(force.detach())

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
            'force_history': torch.cat(force_list),
        }

    def get_physics_params(self):
        """Return all parameters as a dict."""
        return {name: param.item() for name, param in self.params.items()}

    def get_potential_landscape(
        self,
        T: float,
        n_points: int = 200
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the potential landscape for visualization.

        Args:
            T: temperature in C
            n_points: number of evaluation points

        Returns:
            mu_range: (n_points,) COF values
            V_values: (n_points,) potential values
            force_values: (n_points,) force = -dV/dmu values
        """
        device = next(self.parameters()).device
        mu_range = torch.linspace(self.MU_MIN, self.MU_MAX, n_points, device=device)
        T_tensor = torch.tensor(T, dtype=torch.float32, device=device)

        V_values = self.potential(mu_range, T_tensor)
        dVdmu = self.potential_derivative(mu_range, T_tensor)
        force_values = -dVdmu

        return mu_range, V_values, force_values


# --- Verification Script ---
if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Interactive Galling Model V8 - Double-Well Potential Model")
    print("=" * 70)

    model = InteractiveGallingModelV8()

    print("\nLearnable parameters:")
    print(f"  {'#':<4} {'Name':<20} {'Value':>10}")
    print(f"  {'-'*4} {'-'*20} {'-'*10}")
    for i, (name, value) in enumerate(model.get_physics_params().items(), 1):
        print(f"  {i:<4} {name:<20} {value:>10.4f}")
    print(f"\n  Total: {len(model.params)} parameters")

    # --- Plot potential landscapes ---
    temps_landscape = [160, 165, 167.5, 170]
    colors_landscape = {160: '#2ca02c', 165: '#003f5c', 167.5: '#bc5090', 170: '#ffa600'}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('V8 Double-Well Potential Landscapes', fontsize=14)

    with torch.no_grad():
        for idx, temp in enumerate(temps_landscape):
            mu_range, V_values, force_values = model.get_potential_landscape(temp, n_points=300)
            mu_np = mu_range.cpu().numpy()
            V_np = V_values.cpu().numpy()
            F_np = force_values.cpu().numpy()

            color = colors_landscape[temp]

            # Row 0: Potential
            axes[0, idx].plot(mu_np, V_np, color=color, linewidth=2)
            axes[0, idx].set_title(f'{temp} C - V(mu)')
            axes[0, idx].set_ylabel('V(mu)')
            axes[0, idx].set_xlabel('mu')
            axes[0, idx].grid(alpha=0.3)

            # Row 1: Force
            axes[1, idx].plot(mu_np, F_np, color=color, linewidth=2)
            axes[1, idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[1, idx].set_title(f'{temp} C - Force')
            axes[1, idx].set_ylabel('-dV/dmu')
            axes[1, idx].set_xlabel('mu')
            axes[1, idx].grid(alpha=0.3)

            # Check finiteness
            print(f"\n  Potential at {temp} C: min={V_np.min():.4f}, max={V_np.max():.4f}, "
                  f"finite={np.all(np.isfinite(V_np))}")
            print(f"  Force at {temp} C:     min={F_np.min():.4f}, max={F_np.max():.4f}, "
                  f"finite={np.all(np.isfinite(F_np))}")

    plt.tight_layout()
    plt.savefig('v8_potential_landscape.png', dpi=150)
    print("\nSaved: v8_potential_landscape.png")

    # --- Simulate trajectories ---
    temps_sim = [165, 167.5, 170]
    results = {}

    for temp in temps_sim:
        with torch.no_grad():
            res = model.simulate(T=temp, n_cycles=150)
        results[temp] = {k: v.cpu().numpy() for k, v in res.items()}

        mu = results[temp]['mu_history']
        comp = results[temp]['component_history']
        pi = results[temp]['pi_history']
        force = results[temp]['force_history']

        print(f"\n  Simulation at {temp} C:")
        print(f"    COF:    mean={mu.mean():.4f}, std={mu.std():.4f}, "
              f"range=[{mu.min():.4f}, {mu.max():.4f}]")
        print(f"    Jumps:  {comp.sum():.0f}/{len(comp)} ({comp.mean()*100:.1f}%)")
        print(f"    P(stay): mean={pi.mean():.4f}")
        print(f"    Force:  mean={force.mean():.4f}, range=[{force.min():.4f}, {force.max():.4f}]")
        print(f"    Finite: mu={np.all(np.isfinite(mu))}, pi={np.all(np.isfinite(pi))}, "
              f"force={np.all(np.isfinite(force))}")

    # Verify log_prob
    print("\n  Log-prob verification:")
    with torch.no_grad():
        T_test = torch.tensor(167.5)
        mu_test = torch.tensor([0.2, 0.5, 0.8, 1.0])
        dmu_test = torch.tensor([0.01, 0.05, -0.1, 0.0])
        lp = model.log_prob(T_test, mu_test, dmu_test)
        print(f"    mu={mu_test.numpy()}")
        print(f"    dmu={dmu_test.numpy()}")
        print(f"    log_prob={lp.numpy()}")
        print(f"    finite={torch.all(torch.isfinite(lp)).item()}")

    print("\n" + "=" * 70)
    print("V8 verification complete.")
    print("=" * 70)

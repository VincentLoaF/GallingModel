"""
Interactive Galling Model V9 - Fixed-Potential Kramers Galling Model

V9 follows standard Kramers-Smoluchowski theory: the double-well potential
V(mu) is a FIXED material property. Temperature enters ONLY through the
thermal noise intensity D(T), not through the potential itself.

This ensures:
  - Wells always have strong restoring force (no barrier collapse)
  - Physics (potential gradient) explains behavior, not noise
  - Temperature sensitivity via Kramers escape rate: exp(-dV / D(T))

Fixed potential:
    V(mu) = k / L^4 * (mu - mu_L)^2 * (mu - mu_H)^2

where L = mu_H - mu_L, k = material stiffness parameter.

Noise intensity: D(T) = D0 * exp(D_T * dT), dT = T - 160

Derived mixture parameters:
    pi (stay prob)   = clamp(1 - omega0*exp(-dV/D(T)), 0.01, 0.99)
    d1 (stay drift)  = -V'(mu) * dt
    sigma1 (stay noise) = sqrt(2*D(T))
    d2 (jump drift)  = -V'(mu) * tau + j_mu * mu + j_T * dT
    sigma2 (jump noise) = sigma_jump_up if mu < mu_mid else sigma_jump_down

12 learnable parameters, all with physical interpretation.

Date: 2026-02-09
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple


class InteractiveGallingModelV9(nn.Module):
    """
    Fixed-Potential Kramers Galling Model.

    Models P(delta_mu | T, mu) as a 2-component Gaussian mixture where all
    component parameters are derived from a fixed quartic double-well potential
    V(mu). The potential is a material property with no temperature dependence.
    Temperature enters only through thermal noise intensity D(T).
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
                'k': 0.5,                  # Well/barrier stiffness (material property)
                'D0': 0.001,               # Base noise intensity at T_ref
                'D_T': 0.2,               # Temperature scaling of noise
                'omega0': 0.1,             # Escape attempt frequency
                'tau': 3.0,                # Jump time multiplier
                'j_mu': -0.3,              # Jump drift mu-dependence
                'j_T': 0.03,              # Jump drift T-dependence
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
        k = torch.abs(self.params['k'])
        D0 = torch.abs(self.params['D0'])
        D_T = self.params['D_T']
        omega0 = torch.abs(self.params['omega0'])
        tau = torch.abs(self.params['tau'])
        j_mu = self.params['j_mu']
        j_T = self.params['j_T']
        sigma_jump_up = torch.abs(self.params['sigma_jump_up'])
        sigma_jump_down = torch.abs(self.params['sigma_jump_down'])
        return mu_L, mu_H, k, D0, D_T, omega0, tau, j_mu, j_T, sigma_jump_up, sigma_jump_down

    def noise_intensity(self, T: torch.Tensor) -> torch.Tensor:
        """Compute noise intensity D(T) = D0 * exp(D_T * dT)."""
        _, _, _, D0, D_T, _, _, _, _, _, _ = self._get_safe_params()
        dT = T - self.T_REF
        exp_arg = torch.clamp(D_T * dT, -20.0, 20.0)
        return D0 * torch.exp(exp_arg)

    def potential(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Fixed double-well potential V(mu) -- NO temperature dependence.

        V(mu) = k / L^4 * (mu - mu_L)^2 * (mu - mu_H)^2

        This is a material property. Temperature enters only through D(T).
        """
        mu_L, mu_H, k, *_ = self._get_safe_params()
        L = mu_H - mu_L
        L4 = L ** 4 + 1e-8
        V = k / L4 * (mu - mu_L) ** 2 * (mu - mu_H) ** 2
        return V

    def potential_derivative(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Analytical derivative dV/dmu for fixed potential (no tilt).

        dV/dmu = 2k / L^4 * (mu - mu_L) * (mu - mu_H) * (2*mu - mu_L - mu_H)
        """
        mu_L, mu_H, k, *_ = self._get_safe_params()
        L = mu_H - mu_L
        L4 = L ** 4 + 1e-8
        dVdmu = 2.0 * k / L4 * (mu - mu_L) * (mu - mu_H) * (2.0 * mu - mu_L - mu_H)
        return dVdmu

    def well_curvature(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Second derivative V''(mu) for diagnostics.

        V''(mu) = 2k/L^4 * [(2mu - mu_L - mu_H)^2 + 2*(mu-mu_L)*(mu-mu_H)]

        At well minima: V''(mu_L) = V''(mu_H) = 2k/L^2
        At barrier top:  V''(mu_mid) = -k/L^2 (negative = unstable)
        """
        mu_L, mu_H, k, *_ = self._get_safe_params()
        L = mu_H - mu_L
        L4 = L ** 4 + 1e-8
        a = 2.0 * mu - mu_L - mu_H
        d2V = 2.0 * k / L4 * (a ** 2 + 2.0 * (mu - mu_L) * (mu - mu_H))
        return d2V

    def barrier_height(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Effective barrier dV = V(mu_mid) - V(mu), clamped >= 0.

        Since V is fixed (no T), this is purely a function of position mu.
        """
        mu_L = self.params['mu_L']
        mu_H = self.params['mu_H']
        mu_mid = (mu_L + mu_H) / 2.0

        V_mid = self.potential(mu_mid.unsqueeze(0) if mu_mid.dim() == 0 else mu_mid)
        V_mu = self.potential(mu)

        if V_mid.dim() == 0 or (V_mid.shape != V_mu.shape):
            V_mid = V_mid.expand_as(V_mu)

        dV = V_mid - V_mu
        dV = torch.clamp(dV, min=0.0)
        return dV

    def transition_params(self, T: torch.Tensor, mu: torch.Tensor):
        """
        Compute all mixture parameters from fixed potential + thermal noise.

        Returns:
            pi: mixing weight P(stay) in (0.01, 0.99)
            d1: stay component drift
            sigma1: stay component noise
            d2: jump component drift
            sigma2: jump component noise
        """
        mu_L, mu_H, k, D0, D_T, omega0, tau, j_mu, j_T, sigma_jump_up, sigma_jump_down = self._get_safe_params()
        mu_mid = (mu_L + mu_H) / 2.0

        # Noise intensity -- the ONLY T-dependent quantity
        D = self.noise_intensity(T)

        # Barrier height (fixed, no T)
        dV = self.barrier_height(mu)

        # pi (stay probability) via Kramers escape rate
        exp_arg = torch.clamp(-dV / (D + 1e-8), -20.0, 20.0)
        pi = torch.clamp(1.0 - omega0 * torch.exp(exp_arg), 0.01, 0.99)

        # Stay drift: d1 = -V'(mu) * dt (strong restoring force, always present)
        dVdmu = self.potential_derivative(mu)
        d1 = -dVdmu * self.DT

        # Stay noise: sigma1 = sqrt(2 * D(T))
        sigma1 = torch.sqrt(2.0 * D + 1e-8).expand_as(mu)

        # Jump drift: hybrid physics + empirical
        dT = T - self.T_REF
        d2 = -dVdmu * tau + j_mu * mu + j_T * dT

        # Jump noise: asymmetric
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

        log_p1 = self._gaussian_log_prob(delta_mu, d1, sigma1)
        log_p2 = self._gaussian_log_prob(delta_mu, d2, sigma2)

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

        Returns:
            mu_next: next COF (same shape as mu)
            component: which component was sampled (0=stay, 1=jump)
        """
        pi, d1, sigma1, d2, sigma2 = self.transition_params(T, mu)

        is_stay = torch.rand_like(mu) < pi
        component = (~is_stay).float()

        noise = torch.randn_like(mu)
        delta_stay = d1 + sigma1 * noise
        delta_jump = d2 + sigma2 * noise

        delta_mu = torch.where(is_stay, delta_stay, delta_jump)

        mu_next = torch.clamp(mu + delta_mu, self.MU_MIN, self.MU_MAX)

        return mu_next, component

    def get_initial_mu(self, T: float) -> torch.Tensor:
        """Get initial COF for a given temperature."""
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

            # Force = -dV/dmu (restoring force from fixed potential)
            force = -self.potential_derivative(mu)

            pi_list.append(pi.detach())
            d1_list.append(d1.detach())
            d2_list.append(d2.detach())
            sigma1_list.append(sigma1.detach())
            sigma2_list.append(sigma2.detach())
            force_list.append(force.detach())

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

    def get_physics_summary(self) -> dict:
        """Return key derived physics quantities for interpretation."""
        params = self.get_physics_params()
        k = abs(params['k'])
        mu_L = params['mu_L']
        mu_H = params['mu_H']
        D0 = abs(params['D0'])
        D_T = params['D_T']
        L = mu_H - mu_L

        barrier = k / 16.0
        curvature = 2.0 * k / (L ** 2 + 1e-8)

        summary = {
            'barrier_height': barrier,
            'well_curvature': curvature,
            'well_separation': L,
        }

        for dT_val in [0, 5, 7.5, 10]:
            T = 160 + dT_val
            D = D0 * math.exp(D_T * dT_val)
            summary[f'D_at_{T}C'] = D
            summary[f'dV_over_D_at_{T}C'] = barrier / (D + 1e-12)

        return summary

    def get_potential_landscape(
        self,
        T: float,
        n_points: int = 200
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the potential landscape for visualization.

        Note: V(mu) is T-independent. T is accepted for API compatibility
        but does not affect the potential or force.

        Returns:
            mu_range: (n_points,) COF values
            V_values: (n_points,) potential values
            force_values: (n_points,) force = -dV/dmu values
        """
        device = next(self.parameters()).device
        mu_range = torch.linspace(self.MU_MIN, self.MU_MAX, n_points, device=device)

        V_values = self.potential(mu_range)
        dVdmu = self.potential_derivative(mu_range)
        force_values = -dVdmu

        return mu_range, V_values, force_values


# --- Verification Script ---
if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Interactive Galling Model V9 - Fixed-Potential Kramers Model")
    print("=" * 70)

    model = InteractiveGallingModelV9()

    print("\nLearnable parameters:")
    print(f"  {'#':<4} {'Name':<20} {'Value':>10}")
    print(f"  {'-'*4} {'-'*20} {'-'*10}")
    for i, (name, value) in enumerate(model.get_physics_params().items(), 1):
        print(f"  {i:<4} {name:<20} {value:>10.4f}")
    print(f"\n  Total: {len(model.params)} parameters")

    # --- Physics summary ---
    summary = model.get_physics_summary()
    print(f"\n  Barrier height: dV = k/16 = {summary['barrier_height']:.6f}")
    print(f"  Well curvature: V''(well) = {summary['well_curvature']:.4f}")
    print(f"  Well separation: L = {summary['well_separation']:.4f}")
    for T in [160, 165, 167.5, 170]:
        D = summary[f'D_at_{T}C']
        ratio = summary[f'dV_over_D_at_{T}C']
        status = 'stable' if ratio > 5 else 'bistable' if ratio > 2 else 'transitions likely'
        print(f"  {T}C: D={D:.6f}, dV/D={ratio:.2f} ({status})")

    # --- Plot fixed potential landscape ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('V9 Fixed-Potential Kramers Model', fontsize=14)

    with torch.no_grad():
        mu_range, V_values, force_values = model.get_potential_landscape(160, n_points=300)
        mu_np = mu_range.cpu().numpy()
        V_np = V_values.cpu().numpy()
        F_np = force_values.cpu().numpy()

        mu_L = model.params['mu_L'].item()
        mu_H = model.params['mu_H'].item()
        mu_mid = (mu_L + mu_H) / 2.0

        # Panel 1: Fixed potential V(mu)
        axes[0].plot(mu_np, V_np, 'b-', linewidth=2)
        axes[0].axvline(mu_L, color='green', linestyle=':', alpha=0.6, label=f'mu_L={mu_L:.2f}')
        axes[0].axvline(mu_H, color='red', linestyle=':', alpha=0.6, label=f'mu_H={mu_H:.2f}')
        axes[0].set_title('Fixed Potential V(mu)')
        axes[0].set_ylabel('V(mu)')
        axes[0].set_xlabel('mu')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        # Panel 2: Force F(mu) = -dV/dmu
        axes[1].plot(mu_np, F_np, 'b-', linewidth=2)
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].axvline(mu_L, color='green', linestyle=':', alpha=0.6)
        axes[1].axvline(mu_H, color='red', linestyle=':', alpha=0.6)
        axes[1].set_title('Restoring Force -dV/dmu')
        axes[1].set_ylabel('-dV/dmu')
        axes[1].set_xlabel('mu')
        axes[1].grid(alpha=0.3)

        # Panel 3: Escape rate at different temperatures
        temps = [160, 165, 167.5, 170]
        colors = {160: '#2ca02c', 165: '#003f5c', 167.5: '#bc5090', 170: '#ffa600'}
        dV = model.barrier_height(mu_range)
        omega0 = torch.abs(model.params['omega0'])

        for temp in temps:
            T_t = torch.tensor(temp, dtype=torch.float32)
            D = model.noise_intensity(T_t)
            exp_arg = torch.clamp(-dV / (D + 1e-8), -20.0, 20.0)
            escape_rate = omega0 * torch.exp(exp_arg)
            axes[2].plot(mu_np, escape_rate.cpu().numpy(), color=colors[temp],
                        linewidth=1.5, label=f'{temp}C')

        axes[2].set_title('Kramers Escape Rate')
        axes[2].set_ylabel('omega0 * exp(-dV/D(T))')
        axes[2].set_xlabel('mu')
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('v9_potential_landscape.png', dpi=150)
    print("\nSaved: v9_potential_landscape.png")

    # --- Simulate trajectories ---
    temps_sim = [165, 167.5, 170]
    for temp in temps_sim:
        with torch.no_grad():
            res = model.simulate(T=temp, n_cycles=150)
        mu = res['mu_history'].cpu().numpy()
        comp = res['component_history'].cpu().numpy()
        pi = res['pi_history'].cpu().numpy()

        print(f"\n  Simulation at {temp}C:")
        print(f"    COF:    mean={mu.mean():.4f}, std={mu.std():.4f}, "
              f"range=[{mu.min():.4f}, {mu.max():.4f}]")
        print(f"    Jumps:  {comp.sum():.0f}/{len(comp)} ({comp.mean()*100:.1f}%)")
        print(f"    P(stay): mean={pi.mean():.4f}")

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
    print("V9 verification complete.")
    print("=" * 70)

"""
Interactive Galling Model V2 - Competition-Based Dynamics

Key changes from V1:
1. REMOVED state-dependent noise (σ(β) model)
2. ADDED non-linear healing: removal ∝ M^n (n > 1 creates oscillations at high M)
3. ADDED competition intensity: amplified dynamics at transition (β ≈ 0.5)

Physics interpretation:
- Oscillations at high COF emerge from competition dynamics, not noise
- Thick layers (high M) are mechanically unstable → accelerated removal
- Transition zone has aggressive competition → amplified instability

Stability ranking (achieved via physics, not noise):
- Clean state (β≈0): Slow dynamics → stable
- Galled state (β≈1): M^n healing fights back → oscillating limit cycle
- Transition (β≈0.5): Amplified competition → maximum instability

Date: 2026-01-25
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class InteractiveGallingModelV2(nn.Module):
    """
    Physics-Informed Interactive Galling Model V2 - Competition Dynamics

    Key Improvements over V1:
    1. Non-linear healing (M^n) creates natural oscillations at high M
    2. Competition intensity peaks at transition (β ≈ 0.5)
    3. Constant output noise (more interpretable)

    The oscillations emerge from PHYSICS, not from state-dependent noise.
    """

    def __init__(
        self,
        T_ref: float = 165.0,
        mu_low: float = 0.15,
        mu_high: float = 1.0,
        physics_init: Optional[Dict[str, float]] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.register_buffer('T_ref', torch.tensor(T_ref, dtype=torch.float32))
        self.register_buffer('mu_low', torch.tensor(mu_low, dtype=torch.float32))
        self.register_buffer('mu_high', torch.tensor(mu_high, dtype=torch.float32))

        if physics_init is None:
            physics_init = {
                # --- Growth Physics (The Feedback Loop) ---
                'k_adh': 0.05,     # Base adhesion rate
                'alpha_tau': 3.5,  # Shear sensitivity (feedback strength)

                # --- Removal/Healing Physics (The Restoring Force) ---
                'k_spall': 0.8,    # Base spalling rate (healing)
                'E_spall': 0.5,    # Thermal activation energy for sticking

                # --- NEW: Non-linear Healing ---
                'healing_exponent': 1.5,  # n > 1: thick layers are unstable
                                          # Creates oscillations at high M

                # --- NEW: Competition Intensity ---
                'competition_kappa': 2.0,  # κ: amplification at transition
                                           # competition_intensity = 1 + κ·β(1-β)
                                           # Peaks at β = 0.5

                # --- Density Dynamics ---
                'M_sat': 5.0,      # Saturation mass

                # --- Transition Logic ---
                'rho_crit': 0.4,   # Density threshold for regime shift
                'beta_sharpness': 15.0,  # How sharp the transition is

                # --- Stochasticity ---
                'noise_lvl': 0.15,      # Process noise for mass evolution
                'prob_detach': 0.05,    # Probability of massive detachment
                'output_noise': 0.05,   # Constant output noise (SIMPLE!)
            }

        self.physics_params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for name, value in physics_init.items()
        })

    def compute_rates(self, T: torch.Tensor, mu: torch.Tensor, M: torch.Tensor, beta: torch.Tensor):
        """
        Compute competition rates with V2 physics:
        - Non-linear healing (M^n)
        - Competition intensity amplification at transition

        Args:
            T: Temperature
            mu: Current friction coefficient
            M: Current accumulated mass
            beta: Current regime parameter (for competition intensity)
        """
        dT = T - self.T_ref

        # 1. Growth Rate (same as V1)
        softening = torch.exp(0.2 * dT)
        k_growth = self.physics_params['k_adh'] * softening * (mu ** self.physics_params['alpha_tau'])

        # 2. Removal Rate with NON-LINEAR healing
        # Physics: thick layers (high M) are mechanically unstable
        stickiness = torch.exp(self.physics_params['E_spall'] * dT)
        n = self.physics_params['healing_exponent']

        # M^n term: when M is large, removal accelerates super-linearly
        # This creates natural limit cycles at high M
        k_removal_base = self.physics_params['k_spall'] / stickiness

        # Non-linear mass-dependent removal: k_removal * M^n instead of k_removal * M
        # Note: we return the rate, actual removal computed in update_state
        k_removal = k_removal_base

        # 3. Competition Intensity (peaks at β = 0.5)
        # Physics: during transition, neither state dominates → aggressive competition
        kappa = self.physics_params['competition_kappa']
        competition_intensity = 1.0 + kappa * beta * (1 - beta)
        # At β=0: intensity = 1
        # At β=0.5: intensity = 1 + κ/4 (maximum)
        # At β=1: intensity = 1

        return k_growth, k_removal, competition_intensity

    def update_state(self, M: torch.Tensor, rho: torch.Tensor, T: torch.Tensor, mu: torch.Tensor):
        """
        Evolve the physical state with V2 competition dynamics.

        Key V2 changes:
        - dM = competition_intensity * (k_growth - k_removal * M^n)
        - Non-linear M^n creates oscillations at high M
        - Competition intensity amplifies dynamics at transition
        """
        # Compute beta first (needed for competition intensity)
        logit = self.physics_params['beta_sharpness'] * (rho - self.physics_params['rho_crit'])
        beta = torch.sigmoid(logit)

        # Get rates with V2 physics
        k_growth, k_removal, competition_intensity = self.compute_rates(T, mu, M, beta)

        # --- Mass Evolution with V2 dynamics ---
        # Non-linear healing: M^n (n > 1 makes thick layers unstable)
        n = self.physics_params['healing_exponent']
        M_safe = M + 1e-6  # Avoid issues at M=0 for fractional exponents

        # The key V2 formula:
        # dM = competition_intensity * (growth - non_linear_removal) + noise
        growth_term = k_growth
        removal_term = k_removal * (M_safe ** n)

        # Competition intensity amplifies BOTH growth and removal at transition
        dM_deterministic = competition_intensity * (growth_term - removal_term)

        # Process noise
        noise = torch.randn_like(M) * self.physics_params['noise_lvl']
        dM = dM_deterministic + noise

        M_next = torch.clamp(M + dM, min=0.0)

        # --- Massive Detachment Event (same as V1) ---
        p_detach = self.physics_params['prob_detach'] * torch.exp(-(T - self.T_ref) * 0.5)
        is_detach = torch.rand_like(M) < p_detach
        M_next = torch.where(is_detach, M_next * 0.5, M_next)

        # --- Density Evolution ---
        rho_next = torch.tanh(M_next / self.physics_params['M_sat'])

        return M_next, rho_next, is_detach

    def compute_friction(self, rho: torch.Tensor):
        """
        Compute CoF based on surface state with CONSTANT noise.

        V2 simplification: No state-dependent noise!
        Oscillations emerge from competition dynamics, not noise variations.

        mu = (1-beta)*mu_low + beta*mu_high + N(0, σ)
        """
        # Beta is the "Galling Probability"
        logit = self.physics_params['beta_sharpness'] * (rho - self.physics_params['rho_crit'])
        beta = torch.sigmoid(logit)

        # Base friction (Yang's formula)
        mu = (1 - beta) * self.mu_low + beta * self.mu_high

        # CONSTANT output noise (V2 simplification)
        sigma = self.physics_params['output_noise']
        mu_noise = torch.randn_like(mu) * sigma

        return torch.clamp(mu + mu_noise, 0.1, 1.3)

    def simulate_multiple_cycles(
        self,
        T: float,
        n_cycles: int = 150,
        M_init: float = 0.0,
        rho_init: float = 0.0,
        add_noise: bool = True
    ):
        """Simulate multiple cycles with V2 competition dynamics."""
        device = next(self.parameters()).device

        T_tensor = torch.tensor(T, dtype=torch.float32, device=device)

        # Initialize
        M = torch.tensor([M_init], dtype=torch.float32, device=device)
        rho = torch.tensor([rho_init], dtype=torch.float32, device=device)
        mu = self.mu_low.unsqueeze(0)

        mu_list = []
        rho_list = []
        M_list = []
        beta_list = []
        competition_list = []  # Track competition intensity

        for _ in range(n_cycles):
            # 1. Update Physics
            M, rho, _ = self.update_state(M, rho, T_tensor, mu)

            # 2. Update Friction
            mu = self.compute_friction(rho)

            # 3. Compute tracking variables
            logit = self.physics_params['beta_sharpness'] * (rho - self.physics_params['rho_crit'])
            beta = torch.sigmoid(logit)

            kappa = self.physics_params['competition_kappa']
            competition = 1.0 + kappa * beta * (1 - beta)

            mu_list.append(mu)
            rho_list.append(rho)
            M_list.append(M)
            beta_list.append(beta)
            competition_list.append(competition)

        return {
            'mu_history': torch.cat(mu_list),
            'rho_history': torch.cat(rho_list),
            'M_history': torch.cat(M_list),
            'beta_history': torch.cat(beta_list),
            'competition_history': torch.cat(competition_list)  # NEW: track competition
        }

    def get_physics_params(self):
        """Return physics parameters as dict."""
        return {name: param.item() for name, param in self.physics_params.items()}


# --- Verification Script ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    print("=" * 70)
    print("Interactive Galling Model V2 - Competition Dynamics")
    print("=" * 70)

    model = InteractiveGallingModelV2()

    # Print parameters
    print("\nPhysics parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:20s} = {value:.4f}")

    # Simulate all temperatures
    temps = [165, 167.5, 170]
    results = {}

    for temp in temps:
        res = model.simulate_multiple_cycles(T=temp, n_cycles=150)
        results[temp] = {k: v.detach().numpy() for k, v in res.items()}

        mu = results[temp]['mu_history']
        beta = results[temp]['beta_history']
        competition = results[temp]['competition_history']

        print(f"\n{temp}°C:")
        print(f"  COF: mean={mu.mean():.4f}, std={mu.std():.4f}")
        print(f"  β: mean={beta.mean():.4f}, final={beta[-1]:.4f}")
        print(f"  Competition intensity: mean={competition.mean():.4f}, max={competition.max():.4f}")

    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Interactive Galling Model V2 - Competition Dynamics', fontsize=14)

    colors = {165: 'blue', 167.5: 'orange', 170: 'green'}

    for idx, temp in enumerate(temps):
        res = results[temp]
        cycles = np.arange(1, 151)

        # Row 1: COF
        axes[0, idx].plot(cycles, res['mu_history'], color=colors[temp])
        axes[0, idx].set_title(f'{temp}°C - COF')
        axes[0, idx].set_xlabel('Cycle')
        axes[0, idx].set_ylabel('COF')
        axes[0, idx].grid(alpha=0.3)

        # Row 2: Beta (regime)
        axes[1, idx].plot(cycles, res['beta_history'], color=colors[temp])
        axes[1, idx].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[1, idx].set_title(f'{temp}°C - Regime (β)')
        axes[1, idx].set_xlabel('Cycle')
        axes[1, idx].set_ylabel('β')
        axes[1, idx].grid(alpha=0.3)

        # Row 3: Competition Intensity
        axes[2, idx].plot(cycles, res['competition_history'], color=colors[temp])
        axes[2, idx].set_title(f'{temp}°C - Competition Intensity')
        axes[2, idx].set_xlabel('Cycle')
        axes[2, idx].set_ylabel('Intensity')
        axes[2, idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('v2_model_test.png', dpi=150)
    print("\nSaved: v2_model_test.png")
    plt.show()

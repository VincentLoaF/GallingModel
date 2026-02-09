"""
Interactive Galling Model for Multi-Cycle Friction Prediction

Based on Xiao Yang's interactive friction model (Friction 2024) with key improvements:
    μ(t) = (1 - β)·μ_low + β·μ_high

Key innovations:
1. Shear-Driven Growth: Attachment rate depends on Friction (mu), creating feedback loop
2. Temperature-Dependent Spalling: Healing becomes difficult at high T (material softening)
3. Bistability: Sharp transitions between low and high friction states
4. State-Dependent Noise: σ(β) = σ_base + A·β + B·β(1-β) captures stability ranking

Date: 2026-01-16
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class InteractiveGallingModel(nn.Module):
    """
    Physics-Informed Interactive Galling Model (Yang et al. Friction 2024 inspired)

    Key Physics Improvements:
    1. Shear-Driven Growth: Attachment rate depends on Friction (mu), creating the necessary feedback loop.
    2. Temperature-Dependent Spalling: Healing (spalling) becomes difficult at high T (softening).
    3. Critical Mass Bistability: High T lowers the barrier for permanent accumulation.
    """

    def __init__(
        self,
        T_ref: float = 165.0,
        mu_low: float = 0.15,
        mu_high: float = 1.0,  # Adjusted to match data (~1.0 peak)
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
                'alpha_tau': 3.5,  # Shear sensitivity (The "Feedback Strength")
                                   # High alpha means friction spikes cause rapid growth

                # --- Removal/Healing Physics (The Restoring Force) ---
                'k_spall': 0.8,    # Base spalling rate (healing)
                'E_spall': 0.5,    # Thermal activation energy for sticking
                                   # Positive: Higher T = Harder to spall (stickier)

                # --- Density Dynamics ---
                'M_sat': 5.0,      # Saturation mass where rho approaches 1.0

                # --- Transition Logic ---
                'rho_crit': 0.4,   # Density threshold for regime shift
                'beta_sharpness': 15.0, # How sharp the transition is

                # --- Stochasticity (Process Noise) ---
                'noise_lvl': 0.15,   # Random process noise for mass evolution
                'prob_detach': 0.05, # Probability of massive detachment event

                # --- State-Dependent Output Noise ---
                # σ(β) = σ_base + A·β + B·β(1-β)
                # Stability ranking: σ(0) < σ(1) < σ(0.5) requires B > 2A
                'sigma_base': 0.02,       # Baseline noise (clean state, β≈0)
                'sigma_galled': 0.05,     # Additional noise at galled state (A term)
                'sigma_transition': 0.15  # Transition instability peak (B term)
            }

        self.physics_params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for name, value in physics_init.items()
        })

    def compute_rates(self, T: torch.Tensor, mu: torch.Tensor):
        """
        Compute competition rates based on physics.
        """
        # 1. Growth Rate (Attachment)
        # Driven by SHEAR STRESS (mu). This is the key feedback.
        # Physics: Higher T softens material -> easier transfer.
        # Physics: Higher mu -> higher shear stress -> more transfer.
        dT = T - self.T_ref

        # Arrhenius-like softening factor (approximate linear for small range)
        softening = torch.exp(0.2 * dT)

        # Feedback: Growth scales non-linearly with mu
        # If mu is low (0.15), growth is slow. If mu spikes (0.5), growth accelerates.
        k_growth = self.physics_params['k_adh'] * softening * (mu ** self.physics_params['alpha_tau'])

        # 2. Removal Rate (Spalling/Healing)
        # Physics: As T increases, Al becomes sticky/ductile -> Spalling decreases (Healing drops)
        # This causes the "trap" at high temperatures.
        stickiness = torch.exp(self.physics_params['E_spall'] * dT)
        k_removal = self.physics_params['k_spall'] / stickiness

        return k_growth, k_removal

    def update_state(self, M: torch.Tensor, rho: torch.Tensor, T: torch.Tensor, mu: torch.Tensor):
        """
        Evolve the physical state (Mass and Density).
        """
        k_growth, k_removal = self.compute_rates(T, mu)

        # --- Mass Evolution (dM/dn) ---
        # Growth - Removal
        # Add stochastic noise to the growth (process variations)
        noise = torch.randn_like(M) * self.physics_params['noise_lvl']
        dM = k_growth - (k_removal * M) + noise

        M_next = torch.clamp(M + dM, min=0.0)

        # --- Massive Detachment Event (Probabilistic) ---
        # Occasionally, a large chunk breaks off (restoring force)
        # Probability decreases as T increases (sticky)
        p_detach = self.physics_params['prob_detach'] * torch.exp(-(T - self.T_ref)*0.5)
        is_detach = torch.rand_like(M) < p_detach
        M_next = torch.where(is_detach, M_next * 0.5, M_next) # Lose 50% mass

        # --- Density Evolution (Relation to Mass) ---
        # We assume density saturates as Mass increases (Langmuir-like or sigmoid)
        # rho = M / (M + M_sat) or similar.
        # Here we use a sigmoid mapping for smooth saturation.
        rho_next = torch.tanh(M_next / self.physics_params['M_sat'])

        return M_next, rho_next, is_detach

    def compute_friction(self, rho: torch.Tensor):
        """
        Compute CoF based on surface state with state-dependent noise.

        mu = (1-beta)*mu_low + beta*mu_high + noise

        State-dependent noise model:
            σ(β) = σ_base + A·β + B·β(1-β)

        This captures the stability ranking:
            - Clean state (β≈0): Lowest noise (smooth steel-Al interface)
            - Galled state (β≈1): Medium noise (rough Al-on-Al, but stable mean)
            - Transition (β≈0.5): Highest noise (active tearing/re-attachment)
        """
        # Beta is the "Galling Probability" or "Effective Coverage"
        # It transitions sharply when rho crosses a threshold
        logit = self.physics_params['beta_sharpness'] * (rho - self.physics_params['rho_crit'])
        beta = torch.sigmoid(logit)

        # Base friction (Yang's formula)
        mu = (1 - beta) * self.mu_low + beta * self.mu_high

        # STATE-DEPENDENT NOISE
        # σ(β) = σ_base + A·β + B·β(1-β)
        # - σ_base: Clean state noise (minimum)
        # - A·β: Linear increase toward galled state
        # - B·β(1-β): Parabolic hump peaking at β=0.5 (transition instability)
        sigma_base = self.physics_params['sigma_base']
        A = self.physics_params['sigma_galled']
        B = self.physics_params['sigma_transition']

        sigma = sigma_base + A * beta + B * beta * (1 - beta)
        mu_noise = torch.randn_like(mu) * sigma

        return torch.clamp(mu + mu_noise, 0.1, 1.3)

    def simulate_multiple_cycles(self, T: float, n_cycles: int = 150, M_init: float = 0.0, rho_init: float = 0.0, add_noise: bool = True):
        # Get device from model parameters
        device = next(self.parameters()).device

        T_tensor = torch.tensor(T, dtype=torch.float32, device=device)

        # Initialize
        M = torch.tensor([M_init], dtype=torch.float32, device=device)
        rho = torch.tensor([rho_init], dtype=torch.float32, device=device)
        mu = self.mu_low.unsqueeze(0) # Start at low friction

        mu_list = []
        rho_list = []
        M_list = []
        beta_list = []

        for _ in range(n_cycles):
            # 1. Update Physics (Mass & Density)
            # Critical: We use previous mu to drive current growth (Feedback)
            M, rho, _ = self.update_state(M, rho, T_tensor, mu)

            # 2. Update Resulting Friction
            mu = self.compute_friction(rho)

            # 3. Compute beta for tracking
            logit = self.physics_params['beta_sharpness'] * (rho - self.physics_params['rho_crit'])
            beta = torch.sigmoid(logit)

            # Keep as tensors to preserve gradients
            mu_list.append(mu)
            rho_list.append(rho)
            M_list.append(M)
            beta_list.append(beta)

        # Stack to create tensors with gradients
        return {
            'mu_history': torch.cat(mu_list),
            'rho_history': torch.cat(rho_list),
            'M_history': torch.cat(M_list),
            'beta_history': torch.cat(beta_list)
        }

    def get_physics_params(self):
        """Return physics parameters as dict (for trainer compatibility)."""
        return {name: param.item() for name, param in self.physics_params.items()}


# --- Verification Script ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = InteractiveGallingModel()

    # Simulate both conditions
    res_165 = model.simulate_multiple_cycles(T=165.0, n_cycles=150)
    res_170 = model.simulate_multiple_cycles(T=170.0, n_cycles=150)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Trajectories
    ax1.plot(res_165['mu_history'].numpy(), label='165°C (Simulation)', color='blue')
    ax1.plot(res_170['mu_history'].numpy(), label='170°C (Simulation)', color='green')
    ax1.set_title("Simulated Friction Trajectories")
    ax1.set_xlabel("Cycles")
    ax1.set_ylabel("CoF")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Physical State (Mass)
    ax2.plot(res_165['M_history'].numpy(), label='165°C Mass', color='blue', alpha=0.6)
    ax2.plot(res_170['M_history'].numpy(), label='170°C Mass', color='green', alpha=0.6)
    ax2.set_title("Internal State: Transfer Mass (M)")
    ax2.set_xlabel("Cycles")
    ax2.set_ylabel("Mass (arbitrary units)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

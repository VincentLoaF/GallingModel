"""
Interactive Galling Model V5 - Growth + Detachment

V5 replaces V3/V4's probabilistic competition with a physically motivated
three-process model:

1. STOCHASTIC GROWTH: Occurs with T-dependent probability. When it occurs,
   the rate peaks at MID β region (active galling zone), is moderate at HIGH
   (near saturation), and minimal at LOW (clean state).

2. PASSIVE DECAY: Small continuous mass loss every cycle (healing).

3. STOCHASTIC DETACHMENT: Sudden chunk removal. Probability depends on both
   mass (thicker buildup → more fragile) and temperature (hotter → stickier).
   Severity is variable, drawn from Uniform(f_min, f_max).

This creates realistic dynamics:
- LOW state: Growth rare, decay dominates → very stable
- HIGH state: Growth active, frequent detachments → oscillations
- Transition: Uncertain → chaotic bistability

Date: 2026-02-02
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class InteractiveGallingModelV5(nn.Module):
    """
    Physics-Informed Interactive Galling Model V5 - Growth + Detachment

    Key Innovation:
    - Mass accumulates via stochastic growth events (not competition)
    - Mass is removed by stochastic detachment events (not healing competition)
    - Small passive decay provides gentle drift toward clean state
    - Growth rate is state-dependent: peaks at MID β (active galling zone)

    Physical Interpretation:
    - Aluminum transfer builds up gradually during sliding
    - Accumulated material can flake off in chunks (detachment)
    - The combination creates natural oscillations at HIGH state
    - At LOW state, growth is rare → system stays clean and stable
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

        self.KDE_SMOOTHING = 0.005  # Fixed, not learnable — only for KDE stability

        if physics_init is None:
            physics_init = {
                # --- Growth Physics ---
                'k_adh': 0.05,           # Base adhesion rate
                'alpha_tau': 3.5,        # Shear sensitivity (friction feedback)
                'p_growth_base': 0.6,    # Base probability of growth event
                's_peak': 1.5,           # State-dependent scaling peak (at β≈0.5)
                's_base': 0.3,           # Baseline growth scaling

                # --- Passive Decay ---
                'decay_rate': 0.02,      # Small continuous mass loss per cycle

                # --- Detachment Physics ---
                'p_detach_base': 0.08,   # Base detachment probability
                'M_half': 1.0,           # Mass at which detach prob is 50% of max
                'c_stick': 0.3,          # Temperature stickiness factor
                'f_min': 0.3,            # Minimum detachment severity
                'f_max': 0.8,            # Maximum detachment severity

                # --- Density Dynamics ---
                'M_sat': 5.0,            # Saturation mass

                # --- Transition Logic ---
                'rho_crit': 0.4,         # Density threshold for β transition
                'beta_sharpness': 15.0,  # Transition sharpness
            }

        self.physics_params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for name, value in physics_init.items()
        })

    def update_state(
        self,
        M: torch.Tensor,
        rho: torch.Tensor,
        T: torch.Tensor,
        mu: torch.Tensor
    ):
        """
        Evolve physical state via growth + decay + detachment.

        Each cycle:
        1. Compute beta from current rho
        2. Stochastic growth event (T-dependent probability, state-dependent rate)
        3. Passive decay (continuous, every cycle)
        4. Stochastic detachment event (mass+T dependent, variable severity)
        """
        dT = T - self.T_ref

        # 1. Compute beta from current rho
        logit = self.physics_params['beta_sharpness'] * (rho - self.physics_params['rho_crit'])
        beta = torch.sigmoid(logit)

        # 2. GROWTH EVENT (stochastic, state-dependent)
        # P(growth this cycle) depends on temperature
        softening = torch.exp(0.2 * dT)
        p_growth = torch.clamp(
            self.physics_params['p_growth_base'] * softening / (1.0 + softening),
            0.01, 0.99
        ).expand_as(M)
        growth_occurs = torch.rand_like(M) < p_growth

        # Growth rate: peaks at MID beta (active galling zone)
        # S(β) = 4β(1-β) * s_peak + s_base
        s_peak = self.physics_params['s_peak']
        s_base = self.physics_params['s_base']
        state_scale = 4.0 * beta * (1.0 - beta) * s_peak + s_base

        # Growth magnitude when it occurs
        dM_growth = (self.physics_params['k_adh'] * softening
                     * (mu ** self.physics_params['alpha_tau'])
                     * state_scale)
        dM_growth = torch.where(growth_occurs, dM_growth, torch.zeros_like(dM_growth))

        # 3. PASSIVE DECAY (continuous, every cycle)
        dM_decay = -self.physics_params['decay_rate'] * M

        # 4. DETACHMENT EVENT (stochastic, mass+temperature dependent)
        # P(detach) = p0 * g(M) * h(T)
        # g(M) = M / (M + M_half) — more mass → more likely to flake
        # h(T) = exp(-c_stick * dT) — hotter → stickier → less detachment
        g_M = M / (M + self.physics_params['M_half'] + 1e-6)
        h_T = torch.exp(-self.physics_params['c_stick'] * dT)
        p_detach = torch.clamp(
            self.physics_params['p_detach_base'] * g_M * h_T,
            0.0, 0.95
        )
        is_detach = torch.rand_like(M) < p_detach

        # Variable severity: Uniform(f_min, f_max)
        f_min = self.physics_params['f_min']
        f_max = self.physics_params['f_max']
        severity = f_min + torch.rand_like(M) * (f_max - f_min)

        # 5. Apply updates
        M_next = M + dM_growth + dM_decay
        M_next = torch.where(is_detach, M_next * (1.0 - severity), M_next)
        M_next = torch.clamp(M_next, min=0.0)

        # 6. Density evolution
        rho_next = torch.tanh(M_next / self.physics_params['M_sat'])

        return M_next, rho_next, growth_occurs, is_detach, p_growth, p_detach

    def compute_friction(self, rho: torch.Tensor):
        """
        Compute CoF based on surface state.

        V5: NO learnable output noise.
        All variability comes from growth/detachment dynamics.
        Tiny fixed smoothing for KDE numerical stability only.
        """
        logit = self.physics_params['beta_sharpness'] * (rho - self.physics_params['rho_crit'])
        beta = torch.sigmoid(logit)

        # Base friction (Yang's formula)
        mu = (1.0 - beta) * self.mu_low + beta * self.mu_high

        # Fixed tiny smoothing (NOT learnable, NOT physics)
        mu_noise = torch.randn_like(mu) * self.KDE_SMOOTHING

        return torch.clamp(mu + mu_noise, 0.1, 1.3), beta

    def simulate_multiple_cycles(
        self,
        T: float,
        n_cycles: int = 150,
        M_init: float = 0.0,
        rho_init: float = 0.0,
        add_noise: bool = True
    ):
        """Simulate multiple cycles with V5 growth + detachment model."""
        device = next(self.parameters()).device

        T_tensor = torch.tensor(T, dtype=torch.float32, device=device)

        # Initialize
        M = torch.tensor([M_init], dtype=torch.float32, device=device)
        rho = torch.tensor([rho_init], dtype=torch.float32, device=device)
        mu = self.mu_low.unsqueeze(0)

        # History tracking
        mu_list = []
        rho_list = []
        M_list = []
        beta_list = []
        p_growth_list = []
        p_detach_list = []
        growth_event_list = []
        detach_event_list = []

        for _ in range(n_cycles):
            # 1. Update Physics
            M, rho, growth_occurs, is_detach, p_growth, p_detach = self.update_state(
                M, rho, T_tensor, mu
            )

            # 2. Update Friction
            mu, beta = self.compute_friction(rho)

            # Store history
            mu_list.append(mu)
            rho_list.append(rho)
            M_list.append(M)
            beta_list.append(beta)
            p_growth_list.append(p_growth)
            p_detach_list.append(p_detach)
            growth_event_list.append(growth_occurs.float())
            detach_event_list.append(is_detach.float())

        return {
            'mu_history': torch.cat(mu_list),
            'rho_history': torch.cat(rho_list),
            'M_history': torch.cat(M_list),
            'beta_history': torch.cat(beta_list),
            'p_growth_history': torch.cat(p_growth_list),
            'p_detach_history': torch.cat(p_detach_list),
            'growth_events': torch.cat(growth_event_list),
            'detach_events': torch.cat(detach_event_list),
        }

    def get_physics_params(self):
        """Return physics parameters as dict."""
        return {name: param.item() for name, param in self.physics_params.items()}


# --- Verification Script ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    print("=" * 70)
    print("Interactive Galling Model V5 - Growth + Detachment")
    print("=" * 70)

    model = InteractiveGallingModelV5()

    print("\nPhysics parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:25s} = {value:.4f}")

    print(f"\nTotal learnable parameters: {len(model.physics_params)}")

    # Simulate all temperatures
    temps = [165, 167.5, 170]
    results = {}

    for temp in temps:
        res = model.simulate_multiple_cycles(T=temp, n_cycles=150)
        results[temp] = {k: v.detach().numpy() for k, v in res.items()}

        mu = results[temp]['mu_history']
        beta = results[temp]['beta_history']
        mass = results[temp]['M_history']
        growth = results[temp]['growth_events']
        detach = results[temp]['detach_events']

        print(f"\n{temp}°C:")
        print(f"  COF: mean={mu.mean():.4f}, std={mu.std():.4f}")
        print(f"  Mass: mean={mass.mean():.4f}, max={mass.max():.4f}")
        print(f"  Beta: mean={beta.mean():.4f}, final={beta[-1]:.4f}")
        print(f"  Growth events: {growth.sum():.0f}/{len(growth)} ({growth.mean()*100:.1f}%)")
        print(f"  Detach events: {detach.sum():.0f}/{len(detach)} ({detach.mean()*100:.1f}%)")

    # Plotting
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    fig.suptitle('Interactive Galling Model V5 - Growth + Detachment', fontsize=14)

    colors = {165: '#003f5c', 167.5: '#bc5090', 170: '#ffa600'}

    for idx, temp in enumerate(temps):
        res = results[temp]
        cycles = np.arange(1, 151)

        # Row 1: COF
        axes[0, idx].plot(cycles, res['mu_history'], color=colors[temp], linewidth=1)
        axes[0, idx].set_title(f'{temp}°C - COF')
        axes[0, idx].set_ylabel('COF')
        axes[0, idx].set_ylim(0, 1.3)
        axes[0, idx].grid(alpha=0.3)

        # Row 2: Beta (regime)
        axes[1, idx].plot(cycles, res['beta_history'], color=colors[temp])
        axes[1, idx].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[1, idx].set_title(f'{temp}°C - Regime (β)')
        axes[1, idx].set_ylabel('β')
        axes[1, idx].grid(alpha=0.3)

        # Row 3: Mass
        axes[2, idx].plot(cycles, res['M_history'], color=colors[temp])
        axes[2, idx].set_title(f'{temp}°C - Transfer Mass')
        axes[2, idx].set_ylabel('Mass')
        axes[2, idx].grid(alpha=0.3)

        # Row 4: Events (growth + detachment)
        growth_cycles = cycles[res['growth_events'] > 0.5]
        detach_cycles = cycles[res['detach_events'] > 0.5]
        axes[3, idx].scatter(growth_cycles, np.ones_like(growth_cycles),
                            c='green', s=15, alpha=0.7, label='Growth')
        axes[3, idx].scatter(detach_cycles, np.zeros_like(detach_cycles),
                            c='red', s=15, alpha=0.7, label='Detach')
        axes[3, idx].set_title(f'{temp}°C - Events')
        axes[3, idx].set_xlabel('Cycle')
        axes[3, idx].set_ylabel('Event')
        axes[3, idx].set_ylim(-0.3, 1.3)
        axes[3, idx].legend(fontsize=8)
        axes[3, idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('v5_model_test.png', dpi=150)
    print("\nSaved: v5_model_test.png")
    plt.show()

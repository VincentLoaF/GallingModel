"""
Interactive Galling Model V4 - Pure Probabilistic Competition (No Output Noise)

V4 Key Change: Remove output_noise entirely.
All variability comes from probabilistic competition.
A tiny fixed smoothing constant (0.005) is used only for KDE numerical stability.

Key innovation: Oscillations emerge from DISCRETE PROBABILISTIC COMPETITION,
not continuous noise.

Each cycle, there's a competition:
    P(accumulation wins) vs P(healing wins)

The outcome is binary: either growth or removal dominates this cycle.

Probability depends on BOTH:
1. Physical rates (k_growth, k_removal) - thermodynamic driving force
2. State (β) - regime awareness

This naturally captures:
- LOW state (β≈0): Healing almost always wins → very stable
- HIGH state (β≈1): Balanced competition → oscillations
- MID state (β≈0.5): Maximum uncertainty → chaotic transitions

Date: 2026-01-29
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class InteractiveGallingModelV4(nn.Module):
    """
    Physics-Informed Interactive Galling Model V4 - Pure Probabilistic Competition

    V4 Change from V3:
    - Removed output_noise as learnable parameter
    - All variability emerges from probabilistic competition
    - Fixed KDE_SMOOTHING = 0.005 for numerical stability only

    Key Innovation:
    - Discrete probabilistic outcomes instead of continuous noise
    - P(growth) depends on both physical rates AND current state (β)
    - Creates natural stability ranking without explicit noise tuning

    Physical Interpretation:
    - Each cycle is a "battle" between accumulation and healing
    - The winner is determined probabilistically
    - In stable states, one side almost always wins → low variability
    - In contested states, outcomes vary → high variability
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
                'k_adh': 0.05,      # Base adhesion rate
                'alpha_tau': 3.5,   # Shear sensitivity (feedback strength)

                # --- Removal/Healing Physics ---
                'k_spall': 0.8,     # Base spalling rate
                'E_spall': 0.5,     # Thermal activation for sticking

                # --- Probabilistic Competition Parameters ---
                # P(growth) = p_base * state_modifier
                # p_base = k_growth / (k_growth + k_removal)

                'beta_influence': 2.0,    # How much β affects probability
                                          # Higher = more state-dependent

                'low_state_suppression': 0.1,  # P(growth) multiplier when β≈0
                                               # Low value = healing dominates at clean state

                'transition_boost': 1.5,  # Extra randomness at β≈0.5
                                          # Creates instability during transition

                # --- Step Sizes (when growth/healing wins) ---
                'growth_step': 0.3,    # Mass increase when accumulation wins
                'removal_step': 0.2,   # Mass decrease when healing wins

                # --- Density Dynamics ---
                'M_sat': 5.0,          # Saturation mass

                # --- Transition Logic ---
                'rho_crit': 0.4,       # Density threshold
                'beta_sharpness': 15.0,

                # --- Massive Detachment (rare event) ---
                'prob_detach': 0.03,   # Base probability of massive detachment
            }

        self.physics_params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for name, value in physics_init.items()
        })

    def compute_growth_probability(
        self,
        T: torch.Tensor,
        mu: torch.Tensor,
        M: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute probability that accumulation wins this cycle.

        P(growth wins) = p_base * state_modifier

        Where:
        - p_base = k_growth / (k_growth + k_removal*M) [physical competition]
        - state_modifier depends on β [regime awareness]

        State modifier design:
        - At β≈0 (clean): Suppress growth probability → healing dominates
        - At β≈1 (galled): Allow full competition → oscillations
        - At β≈0.5 (transition): Add extra randomness → instability
        """
        dT = T - self.T_ref

        # 1. Physical rates (same as V1/V2)
        softening = torch.exp(0.2 * dT)
        k_growth = self.physics_params['k_adh'] * softening * (mu ** self.physics_params['alpha_tau'])

        stickiness = torch.exp(self.physics_params['E_spall'] * dT)
        k_removal = self.physics_params['k_spall'] / stickiness

        # 2. Base probability from physical competition
        # When growth rate >> removal rate, p_base → 1
        # When removal rate >> growth rate, p_base → 0
        total_rate = k_growth + k_removal * (M + 0.1) + 1e-6
        p_base = k_growth / total_rate

        # 3. State-dependent modifier
        # At β=0: modifier ≈ low_state_suppression (e.g., 0.1)
        # At β=1: modifier ≈ 1.0 (full competition)
        # At β=0.5: extra boost for instability

        low_suppress = self.physics_params['low_state_suppression']
        beta_inf = self.physics_params['beta_influence']
        trans_boost = self.physics_params['transition_boost']

        # Base modifier: interpolates from low_suppress to 1.0 based on β
        # Using β^beta_inf to make it more nonlinear
        base_modifier = low_suppress + (1.0 - low_suppress) * (beta ** beta_inf)

        # Transition instability: peaks at β=0.5
        # 4*β*(1-β) has max value of 1.0 at β=0.5
        transition_factor = 4.0 * beta * (1 - beta)

        # Final modifier: base + transition boost
        # The transition boost adds extra "coin flip" behavior at β≈0.5
        state_modifier = base_modifier * (1.0 + trans_boost * transition_factor)

        # 4. Final probability
        p_growth = torch.clamp(p_base * state_modifier, 0.01, 0.99)

        return p_growth, k_growth, k_removal

    def update_state(
        self,
        M: torch.Tensor,
        rho: torch.Tensor,
        T: torch.Tensor,
        mu: torch.Tensor
    ):
        """
        Evolve the physical state using PROBABILISTIC COMPETITION.

        Each cycle:
        1. Compute P(growth wins) based on physics + state
        2. Draw random outcome
        3. Apply discrete step (growth OR removal)
        """
        # Compute current beta
        logit = self.physics_params['beta_sharpness'] * (rho - self.physics_params['rho_crit'])
        beta = torch.sigmoid(logit)

        # Get growth probability
        p_growth, k_growth, k_removal = self.compute_growth_probability(T, mu, M, beta)

        # --- PROBABILISTIC COMPETITION ---
        # Draw random number and compare to probability
        random_draw = torch.rand_like(M)
        growth_wins = random_draw < p_growth

        # Apply discrete step based on outcome
        growth_step = self.physics_params['growth_step']
        removal_step = self.physics_params['removal_step']

        # Scale steps by physical rates (so temperature still matters)
        dT = T - self.T_ref
        softening = torch.exp(0.1 * dT)  # Milder scaling for steps

        effective_growth = growth_step * softening
        effective_removal = removal_step / softening

        # Apply the outcome
        dM = torch.where(
            growth_wins,
            effective_growth,           # Accumulation won
            -effective_removal * (M + 0.1)  # Healing won (proportional to mass)
        )

        M_next = torch.clamp(M + dM, min=0.0)

        # --- Massive Detachment Event (rare) ---
        # Less likely at high T (sticky)
        p_detach = self.physics_params['prob_detach'] * torch.exp(-dT * 0.3)
        is_detach = torch.rand_like(M) < p_detach
        M_next = torch.where(is_detach, M_next * 0.3, M_next)  # Lose 70% mass

        # --- Density Evolution ---
        rho_next = torch.tanh(M_next / self.physics_params['M_sat'])

        return M_next, rho_next, is_detach, p_growth, growth_wins

    def compute_friction(self, rho: torch.Tensor):
        """
        Compute CoF based on surface state.

        V4: NO learnable output noise.
        All variability comes from probabilistic competition.
        Tiny fixed smoothing for KDE numerical stability only.
        """
        logit = self.physics_params['beta_sharpness'] * (rho - self.physics_params['rho_crit'])
        beta = torch.sigmoid(logit)

        # Base friction (Yang's formula)
        mu = (1 - beta) * self.mu_low + beta * self.mu_high

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
        """Simulate multiple cycles with V4 pure probabilistic competition."""
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
        p_growth_list = []      # Track growth probability
        outcome_list = []       # Track competition outcomes

        for _ in range(n_cycles):
            # 1. Update Physics with probabilistic competition
            M, rho, is_detach, p_growth, growth_wins = self.update_state(M, rho, T_tensor, mu)

            # 2. Update Friction
            mu, beta = self.compute_friction(rho)

            # Store history
            mu_list.append(mu)
            rho_list.append(rho)
            M_list.append(M)
            beta_list.append(beta)
            p_growth_list.append(p_growth)
            outcome_list.append(growth_wins.float())

        return {
            'mu_history': torch.cat(mu_list),
            'rho_history': torch.cat(rho_list),
            'M_history': torch.cat(M_list),
            'beta_history': torch.cat(beta_list),
            'p_growth_history': torch.cat(p_growth_list),
            'outcome_history': torch.cat(outcome_list)
        }

    def get_physics_params(self):
        """Return physics parameters as dict."""
        return {name: param.item() for name, param in self.physics_params.items()}


# --- Verification Script ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    print("=" * 70)
    print("Interactive Galling Model V4 - Pure Probabilistic Competition")
    print("=" * 70)

    model = InteractiveGallingModelV4()

    print("\nPhysics parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:25s} = {value:.4f}")

    # Simulate all temperatures
    temps = [165, 167.5, 170]
    results = {}

    for temp in temps:
        res = model.simulate_multiple_cycles(T=temp, n_cycles=150)
        results[temp] = {k: v.detach().numpy() for k, v in res.items()}

        mu = results[temp]['mu_history']
        beta = results[temp]['beta_history']
        p_growth = results[temp]['p_growth_history']
        outcomes = results[temp]['outcome_history']

        print(f"\n{temp}°C:")
        print(f"  COF: mean={mu.mean():.4f}, std={mu.std():.4f}")
        print(f"  β: mean={beta.mean():.4f}, final={beta[-1]:.4f}")
        print(f"  P(growth): mean={p_growth.mean():.4f}")
        print(f"  Growth wins: {outcomes.sum():.0f}/{len(outcomes)} ({outcomes.mean()*100:.1f}%)")

    # Plotting
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    fig.suptitle('Interactive Galling Model V4 - Pure Probabilistic Competition', fontsize=14)

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

        # Row 3: Growth Probability
        axes[2, idx].plot(cycles, res['p_growth_history'], color=colors[temp])
        axes[2, idx].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[2, idx].set_title(f'{temp}°C - P(growth wins)')
        axes[2, idx].set_ylabel('Probability')
        axes[2, idx].set_ylim(0, 1)
        axes[2, idx].grid(alpha=0.3)

        # Row 4: Competition Outcomes (binary)
        axes[3, idx].scatter(cycles, res['outcome_history'],
                            c=res['outcome_history'], cmap='RdYlGn',
                            s=10, alpha=0.7)
        axes[3, idx].set_title(f'{temp}°C - Outcomes (1=growth, 0=healing)')
        axes[3, idx].set_xlabel('Cycle')
        axes[3, idx].set_ylabel('Winner')
        axes[3, idx].set_ylim(-0.1, 1.1)
        axes[3, idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('v4_model_test.png', dpi=150)
    print("\nSaved: v4_model_test.png")
    plt.show()

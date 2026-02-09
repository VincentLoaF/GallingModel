"""
Physics-Informed Neural Network (PINN) for galling prediction.

Hybrid model combining:
1. Neural network for within-cycle COF(t) prediction
2. Physics model for cycle-to-cycle mass M(n) evolution
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple


class InCycleCOF_NN(nn.Module):
    """
    Neural network for within-cycle COF prediction.

    Predicts COF at each timestep during a slide cycle based on:
    - Transfer layer mass M
    - Temperature T
    - Sliding position and velocity
    - Contact forces
    - Cycle phase
    """

    def __init__(self, hidden_dims: List[int] = [64, 128, 64, 32], dropout: float = 0.1):
        """
        Initialize neural network.

        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        input_dim = 8  # M, T, sliding_dist, velocity, Fx, Fy, Fz, phase

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())  # Smooth activation for physics
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Tensor [..., 8] with columns:
                      [M_norm, T_norm, sliding_dist, velocity, Fx, Fy, Fz, phase]

        Returns:
            cof: Tensor [...] with predicted COF values scaled to [0, 1.3]
        """
        cof = self.network(features)
        cof = torch.sigmoid(cof) * 1.3  # Scale to reasonable COF range [0, 1.3]

        return cof.squeeze(-1)


class GallingPINN(nn.Module):
    """
    Physics-Informed Neural Network for galling prediction.

    Combines:
    1. Neural network for in-cycle COF(t) prediction
    2. Physics model for cycle-to-cycle M(n) evolution

    The neural network learns complex within-cycle dynamics from 125Hz data.
    The physics model enforces mass balance and temperature-dependent attachment/wear.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [64, 128, 64, 32],
        dropout: float = 0.1,
        physics_init: Dict[str, float] = None
    ):
        """
        Initialize PINN model.

        Args:
            hidden_dims: Hidden layer dimensions for neural network
            dropout: Dropout probability
            physics_init: Initial values for physics parameters
                         {k0, T0, kw, M0, Tc, alpha}
        """
        super().__init__()

        # Neural network component
        self.in_cycle_nn = InCycleCOF_NN(hidden_dims, dropout)

        # Physics parameters (learnable)
        if physics_init is None:
            physics_init = {
                'k0': 1.0,      # Baseline attachment rate
                'T0': 25.0,     # Attachment temperature sensitivity (°C)
                'kw': 0.1,      # Wear rate constant
                'M0': 10.0,     # Baseline critical mass
                'Tc': 45.0,     # Critical mass temperature sensitivity (°C)
                'alpha': 0.2    # Retention fraction after detachment
            }

        self.physics_params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for name, value in physics_init.items()
        })

        # Fixed constants
        self.register_buffer('T_ref', torch.tensor(165.0))  # Reference temperature
        self.register_buffer('f_min', torch.tensor(0.15))   # Minimum observed COF
        self.register_buffer('f_max', torch.tensor(1.2))    # Maximum observed COF

    def forward_incycle(
        self,
        M: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict in-cycle COF using neural network.

        Args:
            M: Transfer layer mass (scalar or tensor)
            features: Tensor [batch, time, 8] or [time, 8] with:
                      [M_placeholder, T_norm, sliding_dist, velocity, Fx, Fy, Fz, phase]
                      Note: M_placeholder (features[..., 0]) will be overwritten

        Returns:
            cof_pred: Tensor [batch, time] or [time] predicted COF
        """
        # Update M in features (overwrite placeholder)
        features_updated = features.clone()
        features_updated[..., 0] = M

        # Forward through NN
        cof_pred = self.in_cycle_nn(features_updated)

        return cof_pred

    def physics_update(
        self,
        M_current: torch.Tensor,
        T: float,
        mu_mean: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        Physics-based mass update for cycle transition.

        Implements mass balance model:
        M(n+1) = M(n) + Q_attach(T) - Q_wear(M, μ)

        With galling detachment when M >= M_crit(T).

        Args:
            M_current: Current transfer layer mass
            T: Temperature (°C)
            mu_mean: Mean COF during cycle

        Returns:
            M_next: Updated mass
            detached: Boolean flag indicating if galling detachment occurred
        """
        # Convert temperature to tensor
        T_tensor = torch.tensor(T, dtype=torch.float32, device=M_current.device)

        # Temperature-dependent attachment (Arrhenius-type)
        # Q_attach = k0 * exp((T - T_ref) / T0)
        Q_attach = self.physics_params['k0'] * torch.exp(
            (T_tensor - self.T_ref) / self.physics_params['T0']
        )

        # Wear during sliding (friction-dependent)
        # Higher COF → more wear
        Q_wear = self.physics_params['kw'] * M_current * mu_mean

        # Mass accumulation
        M_next = M_current + Q_attach - Q_wear

        # Critical mass threshold (temperature-dependent)
        # M_crit decreases with temperature
        M_crit = self.physics_params['M0'] * torch.exp(
            -(T_tensor - self.T_ref) / self.physics_params['Tc']
        )

        # Galling detachment event
        # When accumulated aluminum reaches critical mass, partial detachment occurs
        detached = False
        if M_next >= M_crit:
            M_next = self.physics_params['alpha'] * M_next  # Retain fraction alpha
            detached = True

        # Ensure non-negative mass
        M_next = torch.clamp(M_next, min=0.0)

        return M_next, detached

    def forward_multi_cycle(
        self,
        T: float,
        features_list: List[torch.Tensor],
        M_init: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate multiple cycles with coupled NN + physics.

        This is the core PINN forward pass:
        1. For each cycle n:
           a. Use NN to predict COF(t) given M(n)
           b. Aggregate μ_mean(n) from COF(t)
           c. Use physics to compute M(n+1) from M(n) and μ_mean(n)
        2. Return predictions and mass history

        Args:
            T: Temperature (°C, scalar)
            features_list: List of feature tensors [time, 8], one per cycle
            M_init: Initial transfer layer mass

        Returns:
            Dictionary with:
                - 'cof_predicted': List of [time] tensors (predicted COF per cycle)
                - 'M_history': [n_cycles] tensor (mass evolution)
                - 'mu_mean_history': [n_cycles] tensor (mean COF per cycle)
                - 'detachment_events': [n_cycles] tensor (boolean flags)
        """
        n_cycles = len(features_list)
        # Initialize M_current on the same device as model parameters
        device = next(self.parameters()).device
        M_current = torch.tensor(M_init, dtype=torch.float32, device=device)

        cof_predicted = []
        M_history = []
        mu_mean_history = []
        detachment_events = []

        for cycle_idx, features in enumerate(features_list):
            # Predict in-cycle COF
            # features: [time, 8]
            cof_pred = self.forward_incycle(M_current, features)  # [time]

            # Aggregate cycle-averaged COF
            mu_mean = cof_pred.mean()

            # Physics update for next cycle
            M_next, detached = self.physics_update(M_current, T, mu_mean)

            # Store results
            cof_predicted.append(cof_pred)
            M_history.append(M_current)
            mu_mean_history.append(mu_mean)
            detachment_events.append(detached)

            # Update state
            M_current = M_next

        return {
            'cof_predicted': cof_predicted,
            'M_history': torch.stack(M_history),
            'mu_mean_history': torch.stack(mu_mean_history),
            'detachment_events': torch.tensor(detachment_events, dtype=torch.bool, device=device)
        }

    def get_physics_params(self) -> Dict[str, float]:
        """
        Get current physics parameter values.

        Returns:
            Dictionary mapping parameter names to values
        """
        return {name: param.item() for name, param in self.physics_params.items()}

    def set_physics_params(self, params: Dict[str, float]):
        """
        Set physics parameter values.

        Args:
            params: Dictionary mapping parameter names to new values
        """
        for name, value in params.items():
            if name in self.physics_params:
                self.physics_params[name].data = torch.tensor(value, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown physics parameter: {name}")


def test_pinn_model():
    """Test PINN model with dummy data"""
    print("Testing PINN Model...")
    print("=" * 60)

    # Create model
    model = GallingPINN(
        hidden_dims=[64, 128, 64, 32],
        dropout=0.1
    )

    print(f"\n1. Model architecture:")
    print(f"   NN layers: {[64, 128, 64, 32]}")
    print(f"   Physics params: {list(model.physics_params.keys())}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Create dummy data for one cycle
    n_timesteps = 300
    features = torch.randn(n_timesteps, 8)  # Random features

    print(f"\n2. Single cycle forward pass:")
    M_current = torch.tensor(0.5)
    cof_pred = model.forward_incycle(M_current, features)
    print(f"   Input shape: {features.shape}")
    print(f"   Output COF shape: {cof_pred.shape}")
    print(f"   COF range: [{cof_pred.min():.3f}, {cof_pred.max():.3f}]")

    # Test physics update
    mu_mean = cof_pred.mean()
    M_next, detached = model.physics_update(M_current, T=165.0, mu_mean=mu_mean)
    print(f"\n3. Physics update:")
    print(f"   M_current: {M_current:.4f}")
    print(f"   μ_mean: {mu_mean:.4f}")
    print(f"   M_next: {M_next:.4f}")
    print(f"   Detachment: {detached}")

    # Test multi-cycle simulation
    n_cycles = 10
    features_list = [torch.randn(n_timesteps, 8) for _ in range(n_cycles)]

    print(f"\n4. Multi-cycle simulation ({n_cycles} cycles):")
    output = model.forward_multi_cycle(T=167.5, features_list=features_list, M_init=0.0)

    print(f"   COF predictions: {len(output['cof_predicted'])} cycles")
    print(f"   M history shape: {output['M_history'].shape}")
    print(f"   M evolution: {output['M_history'][:5].detach().numpy()}")
    print(f"   μ_mean evolution: {output['mu_mean_history'][:5].detach().numpy()}")
    print(f"   Detachment events: {output['detachment_events'].sum().item()}/{n_cycles}")

    # Test parameter access
    print(f"\n5. Physics parameters:")
    params = model.get_physics_params()
    for name, value in params.items():
        print(f"   {name}: {value:.4f}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_pinn_model()

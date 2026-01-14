"""
Pure Physics-Based Model for Galling Prediction

This model uses ONLY physics equations (no neural network) to predict COF.
It serves as a baseline to evaluate the value of data-driven components.

Key assumptions:
- COF is directly related to transfer layer mass: μ = f(M)
- Mass evolves according to Arrhenius attachment and friction-dependent wear
- No learning from in-cycle temporal patterns

Author: Claude Code
Date: 2026-01-13
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class GallingPhysicsModel(nn.Module):
    """
    Pure physics-based model for galling prediction (no neural network).

    Uses only mechanistic equations with 8 learnable parameters.
    COF is computed from transfer layer mass using a simple friction law.

    This is NOT a PINN - it contains no neural network component.
    """

    def __init__(
        self,
        physics_init: Optional[Dict[str, float]] = None
    ):
        """
        Initialize pure physics model.

        Args:
            physics_init: Initial values for physics parameters
        """
        super().__init__()

        # Physics parameters (all learnable)
        default_params = {
            'k0': 1.0,      # Attachment rate coefficient
            'T0': 25.0,     # Attachment temperature scale (°C)
            'kw': 0.1,      # Wear coefficient
            'M0': 10.0,     # Critical mass baseline
            'Tc': 45.0,     # Critical mass temperature scale (°C)
            'alpha': 0.2,   # Detachment retention fraction
            'mu_base': 0.3, # Base COF (no transfer layer)
            'mu_slope': 0.08, # COF increase per unit mass
        }

        if physics_init:
            default_params.update(physics_init)

        # Learnable physics parameters
        self.physics_params = nn.ParameterDict({
            'k0': nn.Parameter(torch.tensor(default_params['k0'])),
            'T0': nn.Parameter(torch.tensor(default_params['T0'])),
            'kw': nn.Parameter(torch.tensor(default_params['kw'])),
            'M0': nn.Parameter(torch.tensor(default_params['M0'])),
            'Tc': nn.Parameter(torch.tensor(default_params['Tc'])),
            'alpha': nn.Parameter(torch.tensor(default_params['alpha'])),
            'mu_base': nn.Parameter(torch.tensor(default_params['mu_base'])),
            'mu_slope': nn.Parameter(torch.tensor(default_params['mu_slope'])),
        })

        # Fixed constants
        self.register_buffer('T_ref', torch.tensor(165.0))  # Reference temperature
        self.register_buffer('f_min', torch.tensor(0.15))   # Physical bounds
        self.register_buffer('f_max', torch.tensor(1.2))

    def friction_law(self, M: torch.Tensor) -> torch.Tensor:
        """
        Compute COF from transfer layer mass using simple friction law.

        μ = μ_base + μ_slope * M

        This represents the physical principle that more transfer layer
        creates higher friction.

        Args:
            M: Transfer layer mass [batch_size] or scalar

        Returns:
            COF values [batch_size] or scalar
        """
        mu = self.physics_params['mu_base'] + self.physics_params['mu_slope'] * M

        # Clamp to physical bounds
        mu = torch.clamp(mu, min=self.f_min, max=self.f_max)

        return mu

    def physics_update(
        self,
        M_current: torch.Tensor,
        T: float,
        mu_mean: torch.Tensor
    ) -> tuple:
        """
        Update transfer layer mass using physics equations.

        Same equations as PINN models:
        - Arrhenius attachment: Q_attach = k0 * exp((T - T_ref) / T0)
        - Friction-dependent wear: Q_wear = kw * M * μ
        - Mass balance: M_next = M_current + Q_attach - Q_wear
        - Critical threshold: if M >= M_crit, detachment occurs

        Args:
            M_current: Current transfer layer mass
            T: Temperature (°C)
            mu_mean: Mean COF from current cycle

        Returns:
            (M_next, detached): Updated mass and detachment flag
        """
        T_tensor = torch.tensor(T, dtype=torch.float32, device=M_current.device)

        # Attachment (Arrhenius-type)
        Q_attach = self.physics_params['k0'] * torch.exp(
            (T_tensor - self.T_ref) / self.physics_params['T0']
        )

        # Wear (friction-dependent)
        Q_wear = self.physics_params['kw'] * M_current * mu_mean

        # Mass accumulation
        M_next = M_current + Q_attach - Q_wear

        # Critical mass threshold (temperature-dependent)
        M_crit = self.physics_params['M0'] * torch.exp(
            -(T_tensor - self.T_ref) / self.physics_params['Tc']
        )

        # Galling detachment event
        detached = False
        if M_next >= M_crit:
            M_next = self.physics_params['alpha'] * M_next
            detached = True

        # Ensure non-negative mass
        M_next = torch.clamp(M_next, min=0.0)

        return M_next, detached

    def forward_single_cycle(
        self,
        M_init: torch.Tensor,
        T: float,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate a single cycle using pure physics.

        For each timestep:
        1. Use current mass to compute COF via friction law
        2. Mass stays constant within cycle (simplified assumption)
        3. Update mass once at end of cycle based on cycle-averaged COF

        Args:
            M_init: Initial transfer layer mass (scalar)
            T: Temperature (°C)
            features: [seq_len, feature_dim] - not used, but kept for API compatibility

        Returns:
            Dictionary with:
                - cof_predicted: [seq_len] predicted COF trajectory
                - M_final: scalar final mass
                - mu_mean: scalar mean COF
                - detached: boolean detachment flag
        """
        seq_len = features.shape[0]
        device = features.device

        # Convert scalar to tensor if needed
        if isinstance(M_init, (int, float)):
            M_current = torch.tensor(M_init, dtype=torch.float32, device=device)
        else:
            M_current = M_init

        # Compute COF for entire cycle using friction law
        # Simplified: mass is constant during cycle, updates at end
        cof_predicted = self.friction_law(M_current).repeat(seq_len)

        # Cycle-averaged COF
        mu_mean = cof_predicted.mean()

        # Update mass based on cycle-averaged friction
        M_final, detached = self.physics_update(M_current, T, mu_mean)

        return {
            'cof_predicted': cof_predicted,
            'M_final': M_final,
            'mu_mean': mu_mean,
            'detached': detached
        }

    def forward_multi_cycle(
        self,
        T: float,
        features_list: List[torch.Tensor],
        M_init: float = 0.0
    ) -> Dict[str, any]:
        """
        Simulate multiple sequential cycles.

        Mass accumulates across cycles according to physics.

        Args:
            T: Temperature (°C)
            features_list: List of [seq_len, feature_dim] tensors (one per cycle)
            M_init: Initial mass (default: 0.0)

        Returns:
            Dictionary with:
                - cof_predicted: List of [seq_len] tensors
                - M_history: List of mass values
                - mu_mean_history: [n_cycles] mean COF per cycle
                - detachment_events: List of booleans
        """
        device = features_list[0].device
        M_current = torch.tensor(M_init, dtype=torch.float32, device=device)

        cof_predicted = []
        M_history = [M_current.item()]
        mu_mean_list = []
        detachment_events = []

        for features in features_list:
            # Simulate single cycle
            output = self.forward_single_cycle(M_current, T, features)

            cof_predicted.append(output['cof_predicted'])
            mu_mean_list.append(output['mu_mean'].item())
            detachment_events.append(output['detached'])

            # Update mass for next cycle
            M_current = output['M_final']
            M_history.append(M_current.item())

        mu_mean_history = torch.tensor(mu_mean_list, device=device)

        return {
            'cof_predicted': cof_predicted,
            'M_history': M_history,
            'mu_mean_history': mu_mean_history,
            'detachment_events': detachment_events
        }

    def get_physics_params(self) -> Dict[str, float]:
        """Return current physics parameter values"""
        return {
            name: param.item()
            for name, param in self.physics_params.items()
        }

    def set_physics_params(self, params: Dict[str, float]):
        """Set physics parameter values"""
        with torch.no_grad():
            for name, value in params.items():
                if name in self.physics_params:
                    self.physics_params[name].copy_(torch.tensor(value))

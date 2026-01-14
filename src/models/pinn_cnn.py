"""
CNN-Hybrid Physics-Informed Neural Network (PINN) for Galling Prediction

This module implements a CNN-based variant of the PINN model for comparison
with the feedforward baseline. It uses 1D convolutions to capture local temporal
patterns while maintaining the same physics-informed framework.

Key Features:
- 1D Convolutional layers for local temporal pattern capture
- Fewer parameters (~7k vs ~19k feedforward)
- Parallelizable (unlike LSTM)
- Same physics equations as feedforward model

Author: Claude Code
Date: 2026-01-12
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class InCycleCOF_CNN(nn.Module):
    """
    Hybrid CNN + Feedforward neural network for within-cycle COF prediction.

    Combines:
    - 1D Convolutional layers: Capture local temporal patterns (e.g., force spikes)
    - Feedforward layers: Global feature mapping

    Advantages over pure feedforward:
    - Captures temporal correlations within 5-10 timestep windows
    - Still parallelizable (unlike LSTM)
    - Minimal computational overhead (~7k params vs ~19k feedforward)
    """

    def __init__(
        self,
        input_dim: int = 8,
        conv_channels: List[int] = [16, 32],
        kernel_sizes: List[int] = [5, 5],
        fc_hidden: List[int] = [64, 32],
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Number of input features (default 8: M, T, x, v, Fx, Fy, Fz, phase)
            conv_channels: Output channels for each conv layer
            kernel_sizes: Kernel sizes for each conv layer (temporal window)
            fc_hidden: Hidden layer sizes for feedforward part
            dropout: Dropout probability
        """
        super().__init__()

        # 1D Convolutional layers
        conv_layers = []
        in_channels = input_dim

        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # Same padding
                padding_mode='replicate'   # Handle boundaries
            ))
            conv_layers.append(nn.Tanh())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Feedforward layers after convolution
        fc_layers = []
        input_size = conv_channels[-1]  # Output from last conv layer

        for hidden_size in fc_hidden:
            fc_layers.append(nn.Linear(input_size, hidden_size))
            fc_layers.append(nn.Tanh())
            fc_layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        fc_layers.append(nn.Linear(fc_hidden[-1], 1))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN-hybrid network.

        Args:
            features: [batch, time, 8] tensor
                      [M, T, sliding_dist, velocity, Fx, Fy, Fz, phase]

        Returns:
            cof_pred: [batch, time] predicted COF
        """
        # Conv1d expects [batch, channels, time]
        # Transpose from [batch, time, features] to [batch, features, time]
        features_transposed = features.transpose(1, 2)  # [batch, 8, time]

        # 1D Convolution: capture local temporal patterns
        conv_out = self.conv_layers(features_transposed)  # [batch, conv_channels[-1], time]

        # Transpose back to [batch, time, conv_channels[-1]]
        conv_out = conv_out.transpose(1, 2)

        # Pass through feedforward layers
        # Reshape to [batch*time, conv_channels[-1]]
        batch_size, seq_len, conv_features = conv_out.shape
        conv_out_flat = conv_out.reshape(-1, conv_features)

        # FC prediction: [batch*time, 1]
        cof_flat = self.fc_layers(conv_out_flat)

        # Reshape back to [batch, time]
        cof_pred = cof_flat.reshape(batch_size, seq_len)

        # Apply sigmoid to bound output to [0.15, 1.2]
        cof_pred = torch.sigmoid(cof_pred) * (1.2 - 0.15) + 0.15

        return cof_pred


class GallingPINN_CNN(nn.Module):
    """
    CNN-hybrid PINN variant for comparison with feedforward baseline.

    This model uses the same physics equations as the feedforward PINN but
    replaces the feedforward neural network with a CNN-hybrid architecture
    for improved temporal pattern capture.

    Components:
    1. InCycleCOF_CNN: Neural network for within-cycle COF prediction
    2. Physics model: Arrhenius-type mass balance for cycle-to-cycle evolution

    Physics Parameters (learnable):
    - k0: Baseline attachment rate
    - T0: Attachment temperature sensitivity
    - kw: Wear rate constant
    - M0: Baseline critical mass
    - Tc: Critical mass temperature sensitivity
    - alpha: Retention fraction after detachment
    """

    def __init__(
        self,
        conv_channels: List[int] = [16, 32],
        kernel_sizes: List[int] = [5, 5],
        fc_hidden: List[int] = [64, 32],
        dropout: float = 0.1,
        physics_init: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            conv_channels: Output channels for conv layers
            kernel_sizes: Kernel sizes for conv layers
            fc_hidden: Hidden layer sizes for feedforward part
            dropout: Dropout probability
            physics_init: Initial values for physics parameters
        """
        super().__init__()

        # CNN-hybrid neural network
        self.in_cycle_nn = InCycleCOF_CNN(
            input_dim=8,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            fc_hidden=fc_hidden,
            dropout=dropout
        )

        # Physics parameters (same as feedforward version)
        if physics_init is None:
            physics_init = {
                'k0': 1.0,
                'T0': 25.0,
                'kw': 0.1,
                'M0': 10.0,
                'Tc': 45.0,
                'alpha': 0.2
            }

        self.physics_params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for name, value in physics_init.items()
        })

        # Fixed constants (with NEW T_ref = 165°C)
        self.register_buffer('T_ref', torch.tensor(165.0))  # Changed from 200.0
        self.register_buffer('f_min', torch.tensor(0.15))
        self.register_buffer('f_max', torch.tensor(1.2))

    def forward_incycle(
        self,
        M: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict in-cycle COF using CNN-hybrid network.

        Args:
            M: Transfer layer mass (scalar or tensor)
            features: [batch, time, 8] or [time, 8] tensor

        Returns:
            cof_pred: [batch, time] or [time] predicted COF
        """
        # Handle single sequence case
        if features.dim() == 2:
            features = features.unsqueeze(0)  # Add batch dim
            squeeze_output = True
        else:
            squeeze_output = False

        # Update M in features (overwrite placeholder)
        features_updated = features.clone()
        features_updated[..., 0] = M

        # CNN-hybrid forward pass
        cof_pred = self.in_cycle_nn(features_updated)

        if squeeze_output:
            cof_pred = cof_pred.squeeze(0)

        return cof_pred

    def physics_update(
        self,
        M_current: torch.Tensor,
        T: float,
        mu_mean: torch.Tensor
    ) -> tuple:
        """
        Physics-based mass update (same as feedforward version).

        Uses Arrhenius-type temperature dependencies normalized to T_ref=165°C.

        Args:
            M_current: Current transfer layer mass
            T: Temperature (°C)
            mu_mean: Mean COF during cycle

        Returns:
            M_next: Updated mass after cycle
            detached: Boolean indicating if galling detachment occurred
        """
        T_tensor = torch.tensor(T, dtype=torch.float32, device=M_current.device)

        # Attachment (Arrhenius-type, normalized to 165°C)
        Q_attach = self.physics_params['k0'] * torch.exp(
            (T_tensor - self.T_ref) / self.physics_params['T0']
        )

        # Wear (friction-dependent)
        Q_wear = self.physics_params['kw'] * M_current * mu_mean

        # Mass accumulation
        M_next = M_current + Q_attach - Q_wear

        # Critical mass threshold (normalized to 165°C)
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

    def forward_multi_cycle(
        self,
        T: float,
        features_list: List[torch.Tensor],
        M_init: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-cycle simulation with coupled NN + physics.

        Simulates a sequence of cycles at a given temperature, with the neural
        network predicting in-cycle COF and physics updating mass between cycles.

        Args:
            T: Temperature (°C)
            features_list: List of [time, 8] tensors (one per cycle)
            M_init: Initial transfer layer mass

        Returns:
            Dictionary containing:
            - 'cof_predicted': List of [time] tensors (COF for each cycle)
            - 'M_history': [n_cycles] tensor (mass evolution)
            - 'mu_mean_history': [n_cycles] tensor (mean COF per cycle)
            - 'detachment_events': [n_cycles] bool tensor (detachment flags)
        """
        n_cycles = len(features_list)
        device = next(self.parameters()).device
        M_current = torch.tensor(M_init, dtype=torch.float32, device=device)

        cof_predicted = []
        M_history = []
        mu_mean_history = []
        detachment_events = []

        for cycle_idx, features in enumerate(features_list):
            # Predict in-cycle COF using CNN
            cof_pred = self.forward_incycle(M_current, features)

            # Aggregate to cycle-averaged COF
            mu_mean = cof_pred.mean()

            # Physics update for next cycle
            M_next, detached = self.physics_update(M_current, T, mu_mean)

            # Store results
            cof_predicted.append(cof_pred)
            M_history.append(M_current)
            mu_mean_history.append(mu_mean)
            detachment_events.append(detached)

            # Update state for next cycle
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
            Dictionary mapping parameter names to their current values
        """
        return {name: param.item() for name, param in self.physics_params.items()}

"""
Model Implementations for Galling Prediction

Three model variants:
1. Feedforward PINN: Neural network + physics (19,271 params)
2. CNN-Hybrid PINN: 1D CNN + physics (7,479 params)
3. Pure Physics: Mechanistic model only (8 params, no NN)
"""

from .pinn_feedforward import GallingPINN
from .pinn_cnn import GallingPINN_CNN
from .physics_model import GallingPhysicsModel

__all__ = ['GallingPINN', 'GallingPINN_CNN', 'GallingPhysicsModel']

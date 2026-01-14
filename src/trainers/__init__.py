"""
Trainers for PINN Models

- PINNTrainer: For feedforward and CNN models (2-stage training)
- PhysicsOnlyTrainer: For physics-only model (single-stage)
"""

from .trainer_feedforward import PINNTrainer
from .trainer_physics_only import PhysicsOnlyTrainer

__all__ = ['PINNTrainer', 'PhysicsOnlyTrainer']

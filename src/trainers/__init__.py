"""
Trainer implementations for galling prediction models (REDESIGNED 2026-01-15)

Current trainers:
- InteractiveGallingTrainer: For InteractiveGallingModel with likelihood + trend loss
- LikelihoodBasedTrainer: For PhysicsGenerativeModel (deprecated)
- (Coming soon) ForecasterTrainer: For LSTM forecaster (Stage 2)

Deprecated trainers (archived):
- PINNTrainer - see archive/deprecated/
- PhysicsGenerativeTrainer - MSE-based (wrong for stochastic models)

Legacy trainers:
- PhysicsOnlyTrainer - Original physics trainer (still useful for comparison)
"""

from .trainer_interactive_galling import InteractiveGallingTrainer

__all__ = ['InteractiveGallingTrainer']

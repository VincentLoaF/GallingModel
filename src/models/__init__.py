"""
Model Implementations for Galling Prediction (REDESIGNED 2026-01-15)

Current models (correct physics-based approach):
1. InteractiveGallingModel: Yang's interactive friction + galling density (16 params)
   - Best model: captures regime transitions and self-healing
2. PhysicsGenerativeModel: Simpler physics model (deprecated)
3. (Coming soon) GallingForecaster: LSTM for multi-step ahead prediction

Deprecated models (archived - learned trivial μ = F_y/F_z):
- GallingPINN (feedforward) - see archive/deprecated/
- GallingPINN_CNN - see archive/deprecated/

Legacy model (useful for comparison):
- GallingPhysicsModel - Original pure physics
"""

from .interactive_galling_model import InteractiveGallingModel

__all__ = ['InteractiveGallingModel']

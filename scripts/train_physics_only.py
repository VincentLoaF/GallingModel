"""
Train Pure Physics-Only Model

Trains the physics-only model (no neural network) by directly optimizing
physics parameters to match cycle-averaged COF observations.

Usage:
    python scripts/train_physics_only.py
"""

import sys
from pathlib import Path
import yaml
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.physics_model import GallingPhysicsModel
from data_preprocessing import HighFrequencyDataLoader
from trainers.trainer_physics_only import PhysicsOnlyTrainer


def main():
    """Train pure physics model (no neural network)"""

    print("\n" + "="*80)
    print("TRAINING PURE PHYSICS MODEL")
    print("No neural network - only mechanistic equations (8 parameters)")
    print("="*80)

    # Load configuration
    print("\nLoading configuration...")
    with open('config/physics_only.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    data_loader = HighFrequencyDataLoader()
    dataset = data_loader.create_pytorch_dataset([165, 167.5, 170])
    print(f"Total cycles: {len(dataset)}")
    print(f"  165°C: {sum(1 for item in dataset if item['T'] == 165)} cycles")
    print(f"  167.5°C: {sum(1 for item in dataset if item['T'] == 167.5)} cycles")
    print(f"  170°C: {sum(1 for item in dataset if item['T'] == 170)} cycles")

    # Create model
    print("\nInitializing pure physics model...")
    model = GallingPhysicsModel(
        physics_init=config['model']['physics_init']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model architecture:")
    print(f"  Friction law: μ = μ_base + μ_slope × M")
    print(f"  Physics: Arrhenius attachment + friction-dependent wear")
    print(f"\nParameters to optimize:")
    for name in model.physics_params.keys():
        print(f"  - {name}")
    print(f"\nTotal: {total_params} parameters (minimal complexity)")
    print(f"  T_ref: {model.T_ref.item()}°C")

    # Create trainer
    trainer = PhysicsOnlyTrainer(model, config)

    # Train (single-stage, no neural network)
    print("\n" + "="*80)
    print("TRAINING: PHYSICS PARAMETER OPTIMIZATION")
    print("="*80)
    print("Note: No Stage 1 (no neural network to pre-train)")
    print("      Directly optimizing physics parameters\n")

    history = trainer.train(dataset)

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel saved to: {config['output']['model_save_dir']}/")
    print(f"Logs saved to: {config['output']['log_dir']}/")

    print("\nFinal physics parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:10s} = {value:.6f}")

    print("\nModel comparison:")
    print(f"  Physics-Only: {total_params} parameters")
    print(f"  CNN-Hybrid:   7,479 parameters")
    print(f"  Feedforward:  19,271 parameters")
    print("\nPhysics-only is the simplest, most interpretable model!")

    # Generate prediction plots automatically
    print("\n" + "="*80)
    print("GENERATING PREDICTION PLOTS")
    print("="*80)
    try:
        import subprocess
        result = subprocess.run([
            'python', 'scripts/plot_model_predictions.py',
            '--model', 'physics'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Prediction plots generated successfully")
        else:
            print(f"⚠ Warning: Could not generate plots automatically")
            print(f"  Run manually: python scripts/plot_model_predictions.py --model physics")
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots automatically: {e}")
        print(f"  Run manually: python scripts/plot_model_predictions.py --model physics")

    print("\nNext steps:")
    print("  1. View plots in: results/physics/")
    print("  2. Compare all models: python experiments/compare_models.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

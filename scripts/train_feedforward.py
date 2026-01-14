"""
Train Feedforward PINN Model

Simple script to train only the feedforward model.

Usage:
    python scripts/train_feedforward.py
"""

import sys
from pathlib import Path
import yaml
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.pinn_feedforward import GallingPINN
from data_preprocessing import HighFrequencyDataLoader
from trainers.trainer_feedforward import PINNTrainer


def main():
    """Train feedforward PINN model"""

    print("\n" + "="*80)
    print("TRAINING FEEDFORWARD PINN MODEL")
    print("="*80)

    # Load configuration
    print("\nLoading configuration...")
    with open('config/feedforward.yaml', 'r') as f:
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
    print("\nInitializing model...")
    model = GallingPINN(
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
        physics_init=config['model']['physics_init']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    nn_params = sum(p.numel() for p in model.in_cycle_nn.parameters())
    physics_params = sum(p.numel() for p in model.physics_params.values())

    print(f"Model architecture:")
    print(f"  Neural network: {nn_params:,} parameters")
    print(f"  Physics params: {physics_params} parameters")
    print(f"  Total: {total_params:,} parameters")
    print(f"  T_ref: {model.T_ref.item()}°C")

    # Create trainer
    trainer = PINNTrainer(model, config)

    # Stage 1: Pre-train neural network
    print("\n" + "="*80)
    print("STAGE 1: PRE-TRAINING NEURAL NETWORK")
    print("="*80)
    history_s1 = trainer.stage1_pretrain_nn(dataset)

    # Stage 2: Joint physics-informed training
    print("\n" + "="*80)
    print("STAGE 2: JOINT PHYSICS-INFORMED TRAINING")
    print("="*80)
    history_s2 = trainer.stage2_joint_training(dataset)

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel saved to: {config['output']['model_save_dir']}/")
    print(f"Plots saved to: {config['output']['plot_save_dir']}/")
    print(f"Logs saved to: {config['output']['log_dir']}/")

    print("\nFinal physics parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:10s} = {value:.6f}")

    # Generate prediction plots automatically
    print("\n" + "="*80)
    print("GENERATING PREDICTION PLOTS")
    print("="*80)
    try:
        import subprocess
        result = subprocess.run([
            'python', 'scripts/plot_model_predictions.py',
            '--model', 'feedforward'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Prediction plots generated successfully")
        else:
            print(f"⚠ Warning: Could not generate plots automatically")
            print(f"  Run manually: python scripts/plot_model_predictions.py --model feedforward")
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots automatically: {e}")
        print(f"  Run manually: python scripts/plot_model_predictions.py --model feedforward")

    print("\nNext steps:")
    print("  1. View plots in: results/feedforward/")
    print("  2. Compare models: python experiments/compare_models.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Main CLI interface for Galling PINN Model.

Usage:
    python main.py --stage 1 --config config/config.yaml
    python main.py --stage 2 --config config/config.yaml
    python main.py --stage both --config config/config.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.data_loader import HighFrequencyDataLoader
from src.pinn_model import GallingPINN
from src.train import PINNTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description='Train PINN model for galling prediction'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--stage',
        type=str,
        choices=['1', '2', 'both'],
        required=True,
        help='Training stage: 1 (NN pretraining), 2 (joint training), or both'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs from config'
    )

    parser.add_argument(
        '--load-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress'
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 70)
    print("GALLING PINN MODEL - TRAINING")
    print("=" * 70)
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Override config if needed
    if args.epochs is not None:
        if args.stage == '1' or args.stage == 'both':
            config['training']['stage1']['n_epochs'] = args.epochs
        if args.stage == '2' or args.stage == 'both':
            config['training']['stage2']['n_epochs'] = args.epochs

    # Setup reproducibility
    setup_seed(config['seed'], config['deterministic'])
    print(f"Random seed: {config['seed']}")

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    loader = HighFrequencyDataLoader(config['data']['base_path'])
    dataset = loader.create_pytorch_dataset(config['data']['temperatures'])

    stats = loader.get_dataset_statistics(dataset)
    print(f"\nDataset Statistics:")
    print(f"  Total cycles: {stats['n_cycles_total']}")
    print(f"  Total timesteps: {stats['total_timesteps']}")
    print(f"  Avg timesteps/cycle: {stats['avg_timesteps_per_cycle']:.1f}")
    print(f"\n  Cycles per temperature:")
    for temp, count in stats['n_cycles_per_temp'].items():
        print(f"    {temp}°C: {count}")

    # Initialize model
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)

    model = GallingPINN(
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
        physics_init=config['model']['physics_init']
    )

    total_params = sum(p.numel() for p in model.parameters())
    nn_params = sum(p.numel() for p in model.in_cycle_nn.parameters())
    physics_params = sum(p.numel() for p in model.physics_params.parameters())

    print(f"\nModel Architecture:")
    print(f"  NN hidden dims: {config['model']['hidden_dims']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"    - NN parameters: {nn_params:,}")
    print(f"    - Physics parameters: {physics_params}")

    print(f"\nInitial Physics Parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name}: {value:.4f}")

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"\nLoading checkpoint from: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint['epoch']}, stage {checkpoint.get('stage', 'unknown')}")
        print(f"  Loss: {checkpoint['loss']:.6f}")

    # Initialize trainer
    trainer = PINNTrainer(
        model=model,
        config=config,
        device=args.device
    )

    # Training
    if args.stage == '1' or args.stage == 'both':
        # Stage 1: NN pre-training
        history_stage1 = trainer.stage1_pretrain_nn(
            dataset=dataset,
            verbose=args.verbose
        )

        print(f"\nStage 1 Summary:")
        print(f"  Final train loss: {history_stage1['train_loss'][-1]:.6f}")
        print(f"  Final val loss: {history_stage1['val_loss'][-1]:.6f}")
        print(f"  Best val loss: {min(history_stage1['val_loss']):.6f}")

    if args.stage == '2' or args.stage == 'both':
        # Load best Stage 1 model if training both stages
        if args.stage == 'both':
            stage1_path = Path(config['output']['model_save_dir']) / 'best_stage1.pth'
            if stage1_path.exists():
                print(f"\nLoading best Stage 1 model from: {stage1_path}")
                checkpoint = torch.load(stage1_path)
                model.load_state_dict(checkpoint['model_state_dict'])

        # Stage 2: Joint training
        history_stage2 = trainer.stage2_joint_training(
            dataset=dataset,
            verbose=args.verbose
        )

        print(f"\nStage 2 Summary:")
        print(f"  Final total loss: {history_stage2['train_loss'][-1]:.6f}")
        print(f"  Final in-cycle loss: {history_stage2['loss_incycle'][-1]:.6f}")
        print(f"  Final physics loss: {history_stage2['loss_physics'][-1]:.6f}")
        print(f"  Best total loss: {min(history_stage2['train_loss']):.6f}")

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    trainer.save_training_history()
    trainer.save_final_physics_params()

    print("\nTraining complete!")
    print(f"Results saved to: {config['output']['model_save_dir']}")


if __name__ == "__main__":
    main()

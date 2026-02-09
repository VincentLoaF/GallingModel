"""
Visualize Model Predictions vs Observed Data

Plots cycle-averaged COF predictions for any trained model (feedforward, CNN, or physics).

Usage:
    python scripts/plot_model_predictions.py --model feedforward
    python scripts/plot_model_predictions.py --model cnn
    python scripts/plot_model_predictions.py --model physics
    python scripts/plot_model_predictions.py --model feedforward --checkpoint results/feedforward/checkpoint_stage2_epoch50.pth

Author: Claude Code
Date: 2026-01-13
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import HighFrequencyDataLoader
from models.pinn_feedforward import GallingPINN
from models.pinn_cnn import GallingPINN_CNN
from models.physics_model import GallingPhysicsModel


def load_model(model_type, checkpoint_path, device):
    """
    Load trained model from checkpoint.

    Args:
        model_type: 'feedforward', 'cnn', or 'physics'
        checkpoint_path: Path to checkpoint file
        device: torch device

    Returns:
        Loaded model
    """
    print(f"Loading {model_type} model from {checkpoint_path}...")

    # Load config to get architecture params
    config_path = f'config/{model_type}.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create model
    if model_type == 'feedforward':
        model = GallingPINN(
            hidden_dims=config['model']['hidden_dims'],
            dropout=config['model']['dropout'],
            physics_init=config['model']['physics_init']
        )
    elif model_type == 'cnn':
        model = GallingPINN_CNN(
            conv_channels=config['model']['conv_channels'],
            kernel_sizes=config['model']['kernel_sizes'],
            fc_hidden=config['model']['fc_hidden'],
            dropout=config['model']['dropout'],
            physics_init=config['model']['physics_init']
        )
    elif model_type == 'physics':
        model = GallingPhysicsModel(
            physics_init=config['model']['physics_init']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def plot_predictions(model, model_name, data_loader, device, save_path):
    """
    Plot predictions vs observed for all temperatures.

    Args:
        model: Trained model
        model_name: Name for plot title
        data_loader: Data loader instance
        device: torch device
        save_path: Where to save the plot
    """
    print("\nGenerating predictions...")

    model.eval()

    # Collect data for each temperature
    temp_data = {165: {'pred': [], 'obs': [], 'cycles': []},
                 167.5: {'pred': [], 'obs': [], 'cycles': []},
                 170: {'pred': [], 'obs': [], 'cycles': []}}

    # Load all data
    dataset = data_loader.create_pytorch_dataset([165, 167.5, 170])

    # Organize by temperature
    temp_datasets = {165: [], 167.5: [], 170: []}
    for item in dataset:
        temp_datasets[item['T']].append(item)

    with torch.no_grad():
        for temp in [165, 167.5, 170]:
            print(f"  Processing {temp}°C...")
            cycles_data = temp_datasets[temp]

            # Get features for all cycles
            features_list = [item['features'].to(device) for item in cycles_data]

            # Run forward pass through all cycles
            output = model.forward_multi_cycle(temp, features_list, M_init=0.0)

            # Extract cycle-averaged predictions
            mu_mean_pred = output['mu_mean_history'].cpu().numpy()

            # Calculate cycle-averaged observed
            mu_mean_obs = np.array([item['mean_cof_observed'] for item in cycles_data])

            # Store
            temp_data[temp]['pred'] = mu_mean_pred
            temp_data[temp]['obs'] = mu_mean_obs
            temp_data[temp]['cycles'] = np.arange(len(mu_mean_pred))

    # Create figure with subplots
    print("\nCreating plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Model: Predictions vs Observed Data',
                 fontsize=16, fontweight='bold')

    temps = [165, 167.5, 170]
    # Updated color scheme: darker for observed, brighter for predicted
    obs_colors = ['#003f5c', '#7a5195', '#bc5090']  # Dark blue, purple, magenta
    pred_colors = ['#ffa600', '#ef5675', '#58508d']  # Orange, coral, indigo

    # Individual temperature plots
    for idx, temp in enumerate(temps):
        ax = axes[idx // 2, idx % 2]
        data = temp_data[temp]

        # Observed: darker colors, solid line with circles
        ax.plot(data['cycles'], data['obs'], 'o-', alpha=0.7,
                label='Observed', color=obs_colors[idx], markersize=4, linewidth=2)
        # Predicted: brighter colors, dashed line with squares
        ax.plot(data['cycles'], data['pred'], 's--', alpha=0.8,
                label='Predicted', color=pred_colors[idx], markersize=3, linewidth=2)

        ax.set_xlabel('Cycle Number', fontsize=11)
        ax.set_ylabel('Cycle-Averaged COF', fontsize=11)
        ax.set_title(f'Temperature: {temp}°C ({len(data["cycles"])} cycles)',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Calculate R² and RMSE
        residuals = data['obs'] - data['pred']
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data['obs'] - np.mean(data['obs']))**2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals**2))

        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    # Combined plot (all temperatures)
    ax = axes[1, 1]
    scatter_colors = ['#ef5675', '#ffa600', '#58508d']  # Distinct colors for scatter
    for idx, temp in enumerate(temps):
        data = temp_data[temp]
        ax.plot(data['obs'], data['pred'], 'o', alpha=0.7,
                label=f'{temp}°C', color=scatter_colors[idx], markersize=5,
                markeredgewidth=0.5, markeredgecolor='black')

    # Perfect prediction line
    all_obs = np.concatenate([temp_data[t]['obs'] for t in temps])
    min_val, max_val = all_obs.min(), all_obs.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'k--',
            linewidth=2, label='Perfect Fit', alpha=0.5)

    ax.set_xlabel('Observed COF', fontsize=11)
    ax.set_ylabel('Predicted COF', fontsize=11)
    ax.set_title('All Temperatures: Predicted vs Observed',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # Overall R² and RMSE
    all_pred = np.concatenate([temp_data[t]['pred'] for t in temps])
    residuals = all_obs - all_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((all_obs - np.mean(all_obs))**2)
    r2_total = 1 - (ss_res / ss_tot)
    rmse_total = np.sqrt(np.mean(residuals**2))

    ax.text(0.05, 0.95, f'Overall:\nR² = {r2_total:.4f}\nRMSE = {rmse_total:.4f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n✓ Predictions plot saved to: {save_path}')
    plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"PREDICTION SUMMARY - {model_name}")
    print(f"{'='*60}")
    print(f"Overall R²:   {r2_total:.4f}")
    print(f"Overall RMSE: {rmse_total:.4f}")
    print(f"\nPer-temperature R²:")
    for temp in temps:
        data = temp_data[temp]
        residuals = data['obs'] - data['pred']
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data['obs'] - np.mean(data['obs']))**2)
        r2 = 1 - (ss_res / ss_tot)
        print(f"  {temp}°C: {r2:.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Plot model predictions vs observed data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_model_predictions.py --model feedforward
  python scripts/plot_model_predictions.py --model cnn
  python scripts/plot_model_predictions.py --model physics
  python scripts/plot_model_predictions.py --model feedforward --checkpoint results/feedforward/checkpoint_stage2_epoch50.pth
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['feedforward', 'cnn', 'physics'],
        help='Model type to visualize'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (default: results/{model}/best_stage2.pth for PINN, best_model.pth for physics)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for plot (default: results/{model}/predictions_vs_observed.png)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='/root/Documents/GallingModel/data',
        help='Path to data directory (default: /root/Documents/GallingModel/data)'
    )

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Determine checkpoint path
    if args.checkpoint is None:
        if args.model == 'physics':
            args.checkpoint = f'results/{args.model}/best_model.pth'
        else:
            args.checkpoint = f'results/{args.model}/best_stage2.pth'

    # Determine output path
    if args.output is None:
        args.output = f'results/{args.model}/predictions_vs_observed.png'

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"\n❌ Error: Checkpoint not found at {args.checkpoint}")
        print(f"\nPlease train the model first:")
        print(f"  python scripts/train_{args.model}.py")
        sys.exit(1)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    data_loader = HighFrequencyDataLoader(base_path=args.data_path)
    print("✓ Data loaded")

    # Load model
    model = load_model(args.model, args.checkpoint, device)

    # Create plot
    model_names = {
        'feedforward': 'Feedforward PINN',
        'cnn': 'CNN-Hybrid PINN',
        'physics': 'Pure Physics'
    }

    plot_predictions(
        model=model,
        model_name=model_names[args.model],
        data_loader=data_loader,
        device=device,
        save_path=args.output
    )

    print("Done!")


if __name__ == '__main__':
    main()

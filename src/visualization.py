"""
Visualization utilities for PINN model results.

Provides functions to plot training history, predictions, and physics parameters.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple


def plot_training_history(
    history_path: str,
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot training history from JSON file.

    Args:
        history_path: Path to training_history.json
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training History', fontsize=16)

    # Stage 1
    if 'stage1' in history and history['stage1']:
        stage1 = history['stage1']
        epochs1 = range(1, len(stage1['train_loss']) + 1)

        axes[0, 0].plot(epochs1, stage1['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs1, stage1['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('Stage 1: NN Pre-training')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].set_yscale('log')

    # Stage 2 - Total loss
    if 'stage2' in history and history['stage2']:
        stage2 = history['stage2']
        epochs2 = range(1, len(stage2['train_loss']) + 1)

        axes[0, 1].plot(epochs2, stage2['train_loss'], linewidth=2, label='Total')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Stage 2: Joint Training - Total Loss')
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].set_yscale('log')

        # Stage 2 - Component losses
        axes[1, 0].plot(epochs2, stage2['loss_incycle'], label='In-cycle', linewidth=2)
        axes[1, 0].plot(epochs2, stage2['loss_physics'], label='Physics', linewidth=2)
        axes[1, 0].plot(epochs2, stage2['loss_reg'], label='Regularization', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Stage 2: Loss Components')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_yscale('log')

    # Summary statistics
    if 'stage1' in history and history['stage1']:
        stage1 = history['stage1']
        best_val_stage1 = min(stage1['val_loss'])
        final_val_stage1 = stage1['val_loss'][-1]

        summary_text = f"Stage 1:\n"
        summary_text += f"  Best Val Loss: {best_val_stage1:.6f}\n"
        summary_text += f"  Final Val Loss: {final_val_stage1:.6f}\n\n"
    else:
        summary_text = ""

    if 'stage2' in history and history['stage2']:
        stage2 = history['stage2']
        best_total_stage2 = min(stage2['train_loss'])
        final_total_stage2 = stage2['train_loss'][-1]

        summary_text += f"Stage 2:\n"
        summary_text += f"  Best Total Loss: {best_total_stage2:.6f}\n"
        summary_text += f"  Final Total Loss: {final_total_stage2:.6f}\n"
        summary_text += f"  Final In-cycle: {stage2['loss_incycle'][-1]:.6f}\n"
        summary_text += f"  Final Physics: {stage2['loss_physics'][-1]:.6f}"

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_physics_parameters(
    params_path: str,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot fitted physics parameters.

    Args:
        params_path: Path to fitted_parameters.json
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    with open(params_path, 'r') as f:
        params = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Fitted Physics Parameters', fontsize=16)

    param_info = {
        'k0': {'ax': (0, 0), 'label': 'k0 (Attachment Rate)', 'expected': (0.001, 10.0)},
        'T0': {'ax': (0, 1), 'label': 'T0 (Attach Temp Sens, °C)', 'expected': (10.0, 50.0)},
        'kw': {'ax': (0, 2), 'label': 'kw (Wear Rate)', 'expected': (0.01, 0.5)},
        'M0': {'ax': (1, 0), 'label': 'M0 (Critical Mass)', 'expected': (1.0, 100.0)},
        'Tc': {'ax': (1, 1), 'label': 'Tc (Crit Mass Temp Sens, °C)', 'expected': (20.0, 80.0)},
        'alpha': {'ax': (1, 2), 'label': 'alpha (Retention Fraction)', 'expected': (0.1, 0.3)}
    }

    for param_name, info in param_info.items():
        ax = axes[info['ax']]
        value = params.get(param_name, 0.0)
        expected_min, expected_max = info['expected']

        # Bar plot
        colors = ['green' if expected_min <= value <= expected_max else 'orange']
        ax.bar([0], [value], color=colors, alpha=0.7, width=0.5)

        # Expected range
        ax.axhline(expected_min, color='red', linestyle='--', linewidth=1, alpha=0.5,
                   label=f'Expected: [{expected_min}, {expected_max}]')
        ax.axhline(expected_max, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.fill_between([-0.5, 0.5], expected_min, expected_max, color='red', alpha=0.1)

        # Labels
        ax.set_ylabel('Value')
        ax.set_title(info['label'], fontsize=11)
        ax.set_xticks([])
        ax.text(0, value, f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Physics parameters plot saved to {save_path}")

    plt.show()


def plot_predictions_vs_observed(
    predictions: Dict[str, np.ndarray],
    observations: Dict[str, np.ndarray],
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot predicted vs observed COF for sample cycles.

    Args:
        predictions: Dict mapping cycle_id -> predicted COF array
        observations: Dict mapping cycle_id -> observed COF array
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    n_samples = min(3, len(predictions))
    cycle_ids = list(predictions.keys())[:n_samples]

    fig, axes = plt.subplots(1, n_samples, figsize=figsize)
    if n_samples == 1:
        axes = [axes]

    fig.suptitle('Predicted vs Observed COF', fontsize=16)

    for idx, cycle_id in enumerate(cycle_ids):
        pred = predictions[cycle_id]
        obs = observations[cycle_id]

        x = np.arange(len(obs))

        axes[idx].plot(x, obs, label='Observed', linewidth=2, alpha=0.7)
        axes[idx].plot(x, pred, label='Predicted', linewidth=2, linestyle='--', alpha=0.7)

        # RMSE
        rmse = np.sqrt(np.mean((pred - obs) ** 2))
        axes[idx].text(0.05, 0.95, f'RMSE: {rmse:.4f}',
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        axes[idx].set_xlabel('Timestep')
        axes[idx].set_ylabel('COF')
        axes[idx].set_title(f'Cycle {cycle_id}')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")

    plt.show()


def plot_mass_evolution(
    M_history: np.ndarray,
    temps: List[float],
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot transfer layer mass evolution over cycles.

    Args:
        M_history: Array of mass values per cycle
        temps: List of temperatures corresponding to cycles
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Transfer Layer Mass Evolution', fontsize=16)

    cycles = np.arange(1, len(M_history) + 1)

    # Evolution plot
    axes[0].plot(cycles, M_history, linewidth=2, marker='o', markersize=3, alpha=0.7)
    axes[0].set_xlabel('Cycle Number')
    axes[0].set_ylabel('Mass M(n)')
    axes[0].set_title('Mass Accumulation Over Cycles')
    axes[0].grid(alpha=0.3)

    # Distribution by temperature
    unique_temps = sorted(set(temps))
    for temp in unique_temps:
        mask = np.array(temps) == temp
        axes[1].hist(M_history[mask], bins=20, alpha=0.6, label=f'{temp}°C')

    axes[1].set_xlabel('Mass M(n)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Mass Distribution by Temperature')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Mass evolution plot saved to {save_path}")

    plt.show()


def main():
    """Example usage"""
    print("Visualization utilities loaded.")
    print("\nExample usage:")
    print("""
from src.visualization import plot_training_history, plot_physics_parameters

# Plot training history
plot_training_history(
    'results/logs/training_history.json',
    save_path='results/plots/training_history.png'
)

# Plot fitted physics parameters
plot_physics_parameters(
    'results/models/fitted_parameters.json',
    save_path='results/plots/physics_parameters.png'
)
    """)


if __name__ == "__main__":
    main()

"""
Visualize Model Comparison Results

Creates comprehensive comparison plots from comparison experiment results.
Works with both 2-model and 3-model comparisons.

Usage:
    # After running compare_models.py
    python scripts/plot_comparison.py
    python scripts/plot_comparison.py --results results/comparison_all_three.json

Author: Claude Code
Date: 2026-01-13
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import argparse


def plot_three_model_comparison(results_file='results/comparison_all_three.json'):
    """
    Create comprehensive 3-model comparison (Feedforward vs CNN vs Physics).

    Generates a 2x3 grid of plots comparing all three models.
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Three-Model Comparison: Feedforward vs CNN vs Physics (T_ref=165°C)',
                 fontsize=16, fontweight='bold')

    colors = {'feedforward': '#1f77b4', 'cnn': '#2ca02c', 'physics_only': '#ff7f0e'}
    labels = {'feedforward': 'Feedforward', 'cnn': 'CNN-Hybrid', 'physics_only': 'Pure Physics'}

    models = ['feedforward', 'cnn', 'physics_only']

    # ========================================================================
    # Plot 1: Parameter Count
    # ========================================================================
    ax = axes[0, 0]
    params = [results[m]['num_parameters'] for m in models]
    bars = ax.bar([labels[m] for m in models], params,
                   color=[colors[m] for m in models], alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, val in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Number of Parameters', fontsize=11, fontweight='bold')
    ax.set_title('Model Complexity', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # ========================================================================
    # Plot 2: Training Time
    # ========================================================================
    ax = axes[0, 1]
    times = [results[m]['training_time'] for m in models]
    bars = ax.bar([labels[m] for m in models], times,
                   color=[colors[m] for m in models], alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, times):
        height = bar.get_height()
        if val < 60:
            label_text = f'{val:.1f}s'
        else:
            label_text = f'{val/60:.1f}min'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label_text,
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # ========================================================================
    # Plot 3: R² Comparison
    # ========================================================================
    ax = axes[0, 2]
    r2_vals = [results[m]['metrics']['r2_overall'] for m in models]
    bars = ax.bar([labels[m] for m in models], r2_vals,
                   color=[colors[m] for m in models], alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, r2_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax.set_title('Overall Fit Quality', fontsize=12, fontweight='bold')
    ax.set_ylim([max(0.5, min(r2_vals) - 0.1), 1.0])
    ax.grid(axis='y', alpha=0.3)

    # ========================================================================
    # Plot 4: RMSE Comparison
    # ========================================================================
    ax = axes[1, 0]

    rmse_cycle = [results[m]['metrics']['rmse_cycle_avg'] for m in models]

    x = np.arange(len(models))
    width = 0.6

    bars = ax.bar(x, rmse_cycle, width,
                   color=[colors[m] for m in models], alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, rmse_cycle):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in models], fontsize=10, fontweight='bold')
    ax.set_ylabel('Cycle-Averaged RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Accuracy (RMSE)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # ========================================================================
    # Plot 5: Per-Temperature R²
    # ========================================================================
    ax = axes[1, 1]

    temps = [165, 167.5, 170]
    x = np.arange(len(temps))
    width = 0.25

    for i, model in enumerate(models):
        r2_per_temp = [results[model]['metrics']['per_temp_metrics'][str(t)]['r2']
                       for t in temps]
        offset = (i - 1) * width
        ax.bar(x + offset, r2_per_temp, width, label=labels[model],
               color=colors[model], alpha=0.7, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}°C' for t in temps], fontsize=10, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax.set_title('R² by Temperature', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ========================================================================
    # Plot 6: Per-Temperature RMSE
    # ========================================================================
    ax = axes[1, 2]

    for i, model in enumerate(models):
        rmse_per_temp = [results[model]['metrics']['per_temp_metrics'][str(t)]['rmse_cycle_avg']
                         for t in temps]
        offset = (i - 1) * width
        ax.bar(x + offset, rmse_per_temp, width, label=labels[model],
               color=colors[model], alpha=0.7, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}°C' for t in temps], fontsize=10, fontweight='bold')
    ax.set_ylabel('RMSE (Cycle-Averaged)', fontsize=11, fontweight='bold')
    ax.set_title('RMSE by Temperature', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = 'results/comparison_all_three.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Comparison plot saved to: {output_path}')

    plt.close()


def plot_physics_parameters_comparison(results_file='results/comparison_all_three.json'):
    """
    Compare fitted physics parameters across all three models.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract physics parameters
    models = ['feedforward', 'cnn', 'physics_only']
    params_names = ['k0', 'T0', 'kw', 'M0', 'Tc', 'alpha']

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Physics Parameters Comparison', fontsize=16, fontweight='bold')

    colors = {'feedforward': '#1f77b4', 'cnn': '#2ca02c', 'physics_only': '#ff7f0e'}
    labels = {'feedforward': 'Feedforward', 'cnn': 'CNN-Hybrid', 'physics_only': 'Pure Physics'}

    for idx, param_name in enumerate(params_names):
        ax = axes[idx // 3, idx % 3]

        values = [results[m]['physics_params'][param_name] for m in models]

        bars = ax.bar([labels[m] for m in models], values,
                       color=[colors[m] for m in models], alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel(f'{param_name} Value', fontsize=11, fontweight='bold')
        ax.set_title(f'Parameter: {param_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()

    output_path = 'results/physics_parameters_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Physics parameters comparison saved to: {output_path}')

    plt.close()


def print_comparison_summary(results_file='results/comparison_all_three.json'):
    """Print text summary of comparison results."""

    with open(results_file, 'r') as f:
        results = json.load(f)

    models = ['feedforward', 'cnn', 'physics_only']
    labels = {'feedforward': 'Feedforward PINN', 'cnn': 'CNN-Hybrid PINN', 'physics_only': 'Pure Physics'}

    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    print("\n1. MODEL COMPLEXITY:")
    for model in models:
        params = results[model]['num_parameters']
        print(f"   {labels[model]:20s}: {params:>8,} parameters")

    print("\n2. TRAINING TIME:")
    for model in models:
        time_sec = results[model]['training_time']
        if time_sec < 60:
            time_str = f"{time_sec:.1f}s"
        else:
            time_str = f"{time_sec/60:.1f}min ({time_sec:.1f}s)"
        print(f"   {labels[model]:20s}: {time_str}")

    print("\n3. PREDICTION ACCURACY:")
    print(f"   {'Model':20s} {'R²':>10s} {'RMSE':>10s}")
    print(f"   {'-'*20} {'-'*10} {'-'*10}")
    for model in models:
        r2 = results[model]['metrics']['r2_overall']
        rmse = results[model]['metrics']['rmse_cycle_avg']
        print(f"   {labels[model]:20s} {r2:>10.4f} {rmse:>10.4f}")

    print("\n4. PER-TEMPERATURE R²:")
    print(f"   {'Model':20s} {'165°C':>10s} {'167.5°C':>10s} {'170°C':>10s}")
    print(f"   {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for model in models:
        r2_165 = results[model]['metrics']['per_temp_metrics']['165']['r2']
        r2_167 = results[model]['metrics']['per_temp_metrics']['167.5']['r2']
        r2_170 = results[model]['metrics']['per_temp_metrics']['170']['r2']
        print(f"   {labels[model]:20s} {r2_165:>10.4f} {r2_167:>10.4f} {r2_170:>10.4f}")

    print("\n5. PHYSICS PARAMETERS:")
    params_names = ['k0', 'T0', 'kw', 'M0', 'Tc', 'alpha']
    print(f"   {'Parameter':10s} {'Feedforward':>12s} {'CNN-Hybrid':>12s} {'Pure Physics':>12s}")
    print(f"   {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    for param in params_names:
        ff_val = results['feedforward']['physics_params'][param]
        cnn_val = results['cnn']['physics_params'][param]
        phys_val = results['physics_only']['physics_params'][param]
        print(f"   {param:10s} {ff_val:>12.4f} {cnn_val:>12.4f} {phys_val:>12.4f}")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize model comparison results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_comparison.py
  python scripts/plot_comparison.py --results results/comparison_all_three.json
  python scripts/plot_comparison.py --no-params  # Skip physics parameters plot
        """
    )

    parser.add_argument(
        '--results',
        type=str,
        default='results/comparison_all_three.json',
        help='Path to comparison results JSON file'
    )

    parser.add_argument(
        '--no-params',
        action='store_true',
        help='Skip physics parameters comparison plot'
    )

    args = parser.parse_args()

    # Check if results file exists
    if not Path(args.results).exists():
        print(f"\n❌ Error: Results file not found at {args.results}")
        print(f"\nPlease run the comparison experiment first:")
        print(f"  python experiments/compare_models.py")
        return

    print('\nCreating comparison visualizations...\n')

    # Main comparison plot
    plot_three_model_comparison(args.results)

    # Physics parameters comparison
    if not args.no_params:
        plot_physics_parameters_comparison(args.results)

    # Print summary
    print_comparison_summary(args.results)

    print('✓ All plots created successfully!')


if __name__ == '__main__':
    main()

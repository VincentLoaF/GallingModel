"""
Three-Way Architecture Comparison: Feedforward vs CNN vs Physics-Only

This script trains all three architectures and compares:
1. Feedforward PINN (neural network + physics)
2. CNN-Hybrid PINN (1D CNN + physics)
3. Pure Physics-Only (no neural network)

Comparison metrics:
- Prediction accuracy (RMSE, R²)
- Model complexity (parameter count)
- Computational efficiency (training time)
- Physics parameter quality

Author: Claude Code
Date: 2026-01-13
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.pinn_feedforward import GallingPINN  # Feedforward PINN
from models.pinn_cnn import GallingPINN_CNN  # CNN-Hybrid PINN
from models.physics_model import GallingPhysicsModel  # Pure physics (no NN)
from data_preprocessing import HighFrequencyDataLoader
from trainers.trainer_feedforward import PINNTrainer
from trainers.trainer_physics_only import PhysicsOnlyTrainer
import yaml


def evaluate_model(model, dataset, device):
    """
    Evaluate model on full dataset.

    Returns:
        {
            'rmse_incycle': float,
            'rmse_cycle_avg': float,
            'r2_overall': float,
            'per_temp_metrics': dict
        }
    """
    model.eval()

    # Group by temperature
    temp_groups = {165: [], 167.5: [], 170: []}
    for item in dataset:
        temp_groups[item['T']].append(item)

    all_pred = []
    all_obs = []
    all_pred_cycle_avg = []
    all_obs_cycle_avg = []

    per_temp_metrics = {}

    with torch.no_grad():
        for temp, cycles_data in temp_groups.items():
            # Get features for all cycles
            features_list = [item['features'].to(device) for item in cycles_data]

            # Run multi-cycle simulation
            output = model.forward_multi_cycle(temp, features_list, M_init=0.0)

            # In-cycle predictions
            cof_predicted = output['cof_predicted']

            # Cycle-averaged predictions
            mu_mean_pred = output['mu_mean_history'].cpu().numpy()
            mu_mean_obs = np.array([item['mean_cof_observed'] for item in cycles_data])

            # Collect for overall metrics
            all_pred_cycle_avg.append(mu_mean_pred)
            all_obs_cycle_avg.append(mu_mean_obs)

            # In-cycle detailed comparison (only if model provides it)
            for i, (pred, item) in enumerate(zip(cof_predicted, cycles_data)):
                obs = item['target_cof'].to(device)
                all_pred.append(pred.cpu().numpy())
                all_obs.append(obs.cpu().numpy())

            # Per-temperature metrics
            residuals = mu_mean_obs - mu_mean_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((mu_mean_obs - np.mean(mu_mean_obs))**2)
            r2_temp = 1 - (ss_res / ss_tot)
            rmse_temp = np.sqrt(np.mean(residuals**2))

            per_temp_metrics[str(temp)] = {
                'r2': float(r2_temp),
                'rmse_cycle_avg': float(rmse_temp),
                'n_cycles': len(cycles_data)
            }

    # Overall metrics
    all_pred_flat = np.concatenate([p.flatten() for p in all_pred])
    all_obs_flat = np.concatenate([o.flatten() for o in all_obs])
    all_pred_cycle_avg_flat = np.concatenate(all_pred_cycle_avg)
    all_obs_cycle_avg_flat = np.concatenate(all_obs_cycle_avg)

    # In-cycle RMSE
    rmse_incycle = float(np.sqrt(np.mean((all_pred_flat - all_obs_flat)**2)))

    # Cycle-averaged RMSE
    rmse_cycle_avg = float(np.sqrt(np.mean((all_pred_cycle_avg_flat - all_obs_cycle_avg_flat)**2)))

    # Overall R²
    residuals = all_obs_cycle_avg_flat - all_pred_cycle_avg_flat
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((all_obs_cycle_avg_flat - np.mean(all_obs_cycle_avg_flat))**2)
    r2_overall = float(1 - (ss_res / ss_tot))

    return {
        'rmse_incycle': rmse_incycle,
        'rmse_cycle_avg': rmse_cycle_avg,
        'r2_overall': r2_overall,
        'per_temp_metrics': per_temp_metrics
    }


def run_three_way_comparison():
    """Compare all three architectures"""

    print("\n" + "="*80)
    print("THREE-WAY ARCHITECTURE COMPARISON")
    print("1. Feedforward PINN (NN + Physics)")
    print("2. CNN-Hybrid PINN (CNN + Physics)")
    print("3. Pure Physics-Only (No NN)")
    print("="*80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading data...")
    data_loader = HighFrequencyDataLoader(
        base_path='/root/Documents/GallingModel/data'
    )
    dataset = data_loader.create_pytorch_dataset([165, 167.5, 170])
    print(f"Total cycles: {len(dataset)}")

    # Load configs
    with open('config/feedforward.yaml', 'r') as f:
        config_ff = yaml.safe_load(f)

    with open('config/cnn.yaml', 'r') as f:
        config_cnn = yaml.safe_load(f)

    with open('config/physics_only.yaml', 'r') as f:
        config_physics = yaml.safe_load(f)

    results = {}

    # ========================================================================
    # EXPERIMENT 1: Feedforward PINN (T_ref=165°C)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: Feedforward PINN (NN + Physics)")
    print("="*80)

    model_ff = GallingPINN(
        hidden_dims=config_ff['model']['hidden_dims'],
        dropout=config_ff['model']['dropout'],
        physics_init=config_ff['model']['physics_init']
    ).to(device)

    print(f"T_ref = {model_ff.T_ref.item()}°C")
    num_params_ff = sum(p.numel() for p in model_ff.parameters())
    print(f"Total parameters: {num_params_ff:,}")

    trainer_ff = PINNTrainer(model_ff, config_ff)

    start_time = time.time()
    print("\n--- Stage 1: Pre-training Neural Network ---")
    history_ff_s1 = trainer_ff.stage1_pretrain_nn(dataset)
    print("\n--- Stage 2: Joint Physics-Informed Training ---")
    history_ff_s2 = trainer_ff.stage2_joint_training(dataset)
    training_time_ff = time.time() - start_time

    print(f"\nTotal training time: {training_time_ff:.1f}s ({training_time_ff/60:.1f} min)")

    print("\nEvaluating on full dataset...")
    metrics_ff = evaluate_model(model_ff, dataset, device)

    results['feedforward'] = {
        'training_time': training_time_ff,
        'num_parameters': num_params_ff,
        'metrics': metrics_ff,
        'physics_params': model_ff.get_physics_params(),
        'T_ref': model_ff.T_ref.item()
    }

    print(f"\nFeedforward Results:")
    print(f"  R² = {metrics_ff['r2_overall']:.4f}")
    print(f"  RMSE (in-cycle) = {metrics_ff['rmse_incycle']:.4f}")
    print(f"  RMSE (cycle-avg) = {metrics_ff['rmse_cycle_avg']:.4f}")

    # ========================================================================
    # EXPERIMENT 2: CNN-Hybrid PINN (T_ref=165°C)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: CNN-Hybrid PINN (CNN + Physics)")
    print("="*80)

    model_cnn = GallingPINN_CNN(
        conv_channels=config_cnn['model']['conv_channels'],
        kernel_sizes=config_cnn['model']['kernel_sizes'],
        fc_hidden=config_cnn['model']['fc_hidden'],
        dropout=config_cnn['model']['dropout'],
        physics_init=config_cnn['model']['physics_init']
    ).to(device)

    print(f"T_ref = {model_cnn.T_ref.item()}°C")
    num_params_cnn = sum(p.numel() for p in model_cnn.parameters())
    print(f"Total parameters: {num_params_cnn:,}")

    trainer_cnn = PINNTrainer(model_cnn, config_cnn)

    start_time = time.time()
    print("\n--- Stage 1: Pre-training Neural Network ---")
    history_cnn_s1 = trainer_cnn.stage1_pretrain_nn(dataset)
    print("\n--- Stage 2: Joint Physics-Informed Training ---")
    history_cnn_s2 = trainer_cnn.stage2_joint_training(dataset)
    training_time_cnn = time.time() - start_time

    print(f"\nTotal training time: {training_time_cnn:.1f}s ({training_time_cnn/60:.1f} min)")

    print("\nEvaluating on full dataset...")
    metrics_cnn = evaluate_model(model_cnn, dataset, device)

    results['cnn'] = {
        'training_time': training_time_cnn,
        'num_parameters': num_params_cnn,
        'metrics': metrics_cnn,
        'physics_params': model_cnn.get_physics_params(),
        'T_ref': model_cnn.T_ref.item()
    }

    print(f"\nCNN-Hybrid Results:")
    print(f"  R² = {metrics_cnn['r2_overall']:.4f}")
    print(f"  RMSE (in-cycle) = {metrics_cnn['rmse_incycle']:.4f}")
    print(f"  RMSE (cycle-avg) = {metrics_cnn['rmse_cycle_avg']:.4f}")

    # ========================================================================
    # EXPERIMENT 3: Pure Physics-Only Model
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 3: Pure Physics Model (No Neural Network)")
    print("="*80)

    model_physics = GallingPhysicsModel(
        physics_init=config_physics['model']['physics_init']
    ).to(device)

    print(f"T_ref = {model_physics.T_ref.item()}°C")
    num_params_physics = sum(p.numel() for p in model_physics.parameters())
    print(f"Total parameters: {num_params_physics:,}")

    trainer_physics = PhysicsOnlyTrainer(model_physics, config_physics)

    start_time = time.time()
    history_physics = trainer_physics.train(dataset)
    training_time_physics = time.time() - start_time

    print(f"\nTotal training time: {training_time_physics:.1f}s ({training_time_physics/60:.1f} min)")

    print("\nEvaluating on full dataset...")
    metrics_physics = evaluate_model(model_physics, dataset, device)

    results['physics_only'] = {
        'training_time': training_time_physics,
        'num_parameters': num_params_physics,
        'metrics': metrics_physics,
        'physics_params': model_physics.get_physics_params(),
        'T_ref': model_physics.T_ref.item()
    }

    print(f"\nPhysics-Only Results:")
    print(f"  R² = {metrics_physics['r2_overall']:.4f}")
    print(f"  RMSE (in-cycle) = {metrics_physics['rmse_incycle']:.4f}")
    print(f"  RMSE (cycle-avg) = {metrics_physics['rmse_cycle_avg']:.4f}")

    # ========================================================================
    # THREE-WAY COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("THREE-WAY COMPARISON SUMMARY")
    print("="*80)

    print("\n1. MODEL COMPLEXITY:")
    print(f"   {'Model':20s} {'Parameters':>12s} {'vs Feedforward':>15s}")
    print(f"   {'-'*20} {'-'*12} {'-'*15}")

    models_info = [
        ('Feedforward', results['feedforward']['num_parameters']),
        ('CNN-Hybrid', results['cnn']['num_parameters']),
        ('Physics-Only', results['physics_only']['num_parameters'])
    ]

    for name, params in models_info:
        reduction = (1 - params/results['feedforward']['num_parameters'])*100
        print(f"   {name:20s} {params:12,} {reduction:+14.1f}%")

    print("\n2. TRAINING TIME:")
    print(f"   {'Model':20s} {'Time (s)':>12s} {'vs Feedforward':>15s}")
    print(f"   {'-'*20} {'-'*12} {'-'*15}")

    for key, name in [('feedforward', 'Feedforward'), ('cnn', 'CNN-Hybrid'), ('physics_only', 'Physics-Only')]:
        time_s = results[key]['training_time']
        ratio = time_s / results['feedforward']['training_time']
        print(f"   {name:20s} {time_s:12.1f} {ratio:14.2f}x")

    print("\n3. PREDICTION ACCURACY:")
    print(f"   {'Metric':20s} {'Feedforward':>12s} {'CNN-Hybrid':>12s} {'Physics-Only':>12s}")
    print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

    metrics_to_compare = [
        ('R² Overall', 'r2_overall'),
        ('RMSE In-Cycle', 'rmse_incycle'),
        ('RMSE Cycle-Avg', 'rmse_cycle_avg')
    ]

    for label, key in metrics_to_compare:
        ff_val = results['feedforward']['metrics'][key]
        cnn_val = results['cnn']['metrics'][key]
        phys_val = results['physics_only']['metrics'][key]
        print(f"   {label:20s} {ff_val:12.4f} {cnn_val:12.4f} {phys_val:12.4f}")

    print("\n4. BEST PERFORMING MODEL:")
    r2_values = {
        'Feedforward': results['feedforward']['metrics']['r2_overall'],
        'CNN-Hybrid': results['cnn']['metrics']['r2_overall'],
        'Physics-Only': results['physics_only']['metrics']['r2_overall']
    }
    best_model = max(r2_values.items(), key=lambda x: x[1])
    print(f"   {best_model[0]} (R² = {best_model[1]:.4f})")

    print("\n5. KEY INSIGHTS:")

    # Compare NN models vs physics-only
    ff_r2 = results['feedforward']['metrics']['r2_overall']
    phys_r2 = results['physics_only']['metrics']['r2_overall']
    improvement = ((ff_r2 - phys_r2) / phys_r2) * 100

    print(f"   - Neural network improves R² by {improvement:.1f}% over pure physics")
    print(f"   - Physics-only has {results['physics_only']['num_parameters']:,} parameters (minimal complexity)")
    print(f"   - {best_model[0]} provides best overall performance")

    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    results_serializable = make_serializable(results)

    with open('results/comparison_all_three.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print("\n" + "="*80)
    print("Results saved to: results/comparison_all_three.json")
    print("="*80)

    # Generate comparison plots automatically
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    try:
        import subprocess
        result = subprocess.run([
            'python', 'scripts/plot_comparison.py'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Comparison plots generated successfully")
            print("  - results/comparison_all_three.png")
            print("  - results/physics_parameters_comparison.png")
        else:
            print(f"⚠ Warning: Could not generate plots automatically")
            print(f"  Run manually: python scripts/plot_comparison.py")
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots automatically: {e}")
        print(f"  Run manually: python scripts/plot_comparison.py")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nView results:")
    print("  - JSON: results/comparison_all_three.json")
    print("  - Plots: results/comparison_all_three.png")
    print("  - Physics params: results/physics_parameters_comparison.png")
    print("="*80 + "\n")

    return results


if __name__ == '__main__':
    results = run_three_way_comparison()

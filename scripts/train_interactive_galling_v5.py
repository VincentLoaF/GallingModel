"""
Training Script for Interactive Galling Model V5 - Growth + Detachment

V5 replaces probabilistic competition with three physical processes:
1. Stochastic growth (T-dependent probability, state-dependent rate)
2. Passive decay (small continuous mass loss)
3. Stochastic detachment (mass+T dependent, variable severity)

Usage:
    python scripts/train_interactive_galling_v5.py                    # Train from scratch
    python scripts/train_interactive_galling_v5.py --resume-from-best # Continue from best
    python scripts/train_interactive_galling_v5.py --validate-only    # Only validate

Author: Claude
Date: 2026-02-02
"""

import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.cycle_averaged_loader import CycleAveragedLoader
from src.models.interactive_galling_model_v5 import InteractiveGallingModelV5
from src.trainers.trainer_interactive_galling import InteractiveGallingTrainer, validate_model


def main():
    parser = argparse.ArgumentParser(description='Train Interactive Galling Model V5')
    parser.add_argument('--resume-from-best', action='store_true',
                        help='Resume training from best checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                        help='Skip training and only run validation')
    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING INTERACTIVE GALLING MODEL V5 - GROWTH + DETACHMENT")
    print("=" * 80)
    print("\nV5 Key Change:")
    print("  Replace probabilistic competition with growth + detachment")
    print("\nV5 Three Processes:")
    print("  1. Stochastic growth (T-dependent prob, state-dependent rate)")
    print("  2. Passive decay (small continuous mass loss)")
    print("  3. Stochastic detachment (mass+T dependent, variable severity)")
    print("=" * 80)

    # Configuration
    config = {
        'learning_rate': 0.005,
        'weight_decay': 1e-5,
        'n_epochs': 200,
        'n_likelihood_samples': 30,
        'kde_bandwidth': 0.08,
        'save_dir': 'results/interactive_galling_v5'
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if args.validate_only:
        print("\n[VALIDATION-ONLY MODE]")
    elif args.resume_from_best:
        print("\n[FINE-TUNING MODE]")

    # ======================================================================
    # 1. LOAD DATA
    # ======================================================================
    print("\n" + "-" * 80)
    print("Loading cycle-averaged COF data...")
    print("-" * 80)

    loader = CycleAveragedLoader()
    temp_data_dict = loader.load_all_temperatures([165, 167.5, 170])

    print("\nDataset summary:")
    for temp, data in sorted(temp_data_dict.items()):
        print(f"  {temp}°C: {data['n_cycles']} cycles")
        print(f"    Mean COF: {data['mean_cof'].mean():.4f} +/- {data['mean_cof'].std():.4f}")

    # ======================================================================
    # 2. CREATE V5 MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Creating Interactive Galling Model V5...")
    print("-" * 80)

    physics_init = {
        # Growth Physics
        'k_adh': 0.05,
        'alpha_tau': 3.5,
        'p_growth_base': 0.6,
        's_peak': 1.5,
        's_base': 0.3,

        # Passive Decay
        'decay_rate': 0.02,

        # Detachment Physics
        'p_detach_base': 0.08,
        'M_half': 1.0,
        'c_stick': 0.3,
        'f_min': 0.3,
        'f_max': 0.8,

        # Density Dynamics
        'M_sat': 5.0,

        # Transition Logic
        'rho_crit': 0.4,
        'beta_sharpness': 15.0,
    }

    model = InteractiveGallingModelV5(
        T_ref=165.0,
        mu_low=0.15,
        mu_high=1.0,
        physics_init=physics_init,
        device=device
    )

    # Load checkpoint if requested
    if args.validate_only or args.resume_from_best:
        checkpoint_path = Path(config['save_dir']) / 'best_model.pth'
        if checkpoint_path.exists():
            print(f"\nLoading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            previous_ll = checkpoint.get('best_log_likelihood',
                                         checkpoint.get('log_likelihood', 'N/A'))
            print(f"  Previous log-likelihood: {previous_ll}")
        else:
            print(f"\nWARNING: Checkpoint not found at {checkpoint_path}")
            if args.validate_only:
                return

    print("\nInitial physics parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:25s} = {value:.4f}")

    # Skip to validation if requested
    if args.validate_only:
        print("\nSkipping training - proceeding to validation...")
        validate_model(
            model=model,
            temp_data_dict=temp_data_dict,
            n_simulations=20,
            save_dir='results/interactive_galling_v5/validation',
            device=device
        )
        return

    # ======================================================================
    # 3. PRE-TRAINING CHECK
    # ======================================================================
    print("\n" + "-" * 80)
    print("Pre-training simulation check...")
    print("-" * 80)

    for temp in [165, 167.5, 170]:
        with torch.no_grad():
            sim = model.simulate_multiple_cycles(T=temp, n_cycles=100)

        mu = sim['mu_history'].cpu().numpy()
        beta = sim['beta_history'].cpu().numpy()
        mass = sim['M_history'].cpu().numpy()
        growth = sim['growth_events'].cpu().numpy()
        detach = sim['detach_events'].cpu().numpy()

        print(f"\n{temp}°C (100 cycles):")
        print(f"  COF: mean={mu.mean():.4f}, std={mu.std():.4f}")
        print(f"  Mass: mean={mass.mean():.4f}, max={mass.max():.4f}")
        print(f"  Beta: mean={beta.mean():.4f}, final={beta[-1]:.4f}")
        print(f"  Growth events: {growth.sum():.0f}/100 ({growth.mean()*100:.1f}%)")
        print(f"  Detach events: {detach.sum():.0f}/100 ({detach.mean()*100:.1f}%)")

    # ======================================================================
    # 4. TRAIN MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Training V5 model (likelihood-based)...")
    print("-" * 80)

    trainer = InteractiveGallingTrainer(model, config, device=device)

    trainer.train(
        temp_data_dict,
        n_epochs=config['n_epochs'],
        patience_limit=30,
        save_dir=config['save_dir'],
        verbose=True
    )

    # ======================================================================
    # 5. VALIDATE MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Validating trained V5 model...")
    print("-" * 80)

    validate_model(
        model=model,
        temp_data_dict=temp_data_dict,
        n_simulations=20,
        save_dir='results/interactive_galling_v5/validation',
        device=device
    )

    # ======================================================================
    # 6. COMPUTE FINAL METRICS
    # ======================================================================
    print("\n" + "-" * 80)
    print("Computing final metrics...")
    print("-" * 80)

    results = {}
    for temp, data in temp_data_dict.items():
        observed = data['mean_cof']
        n_cycles = data['n_cycles']

        with torch.no_grad():
            sim = model.simulate_multiple_cycles(T=temp, n_cycles=n_cycles, add_noise=False)

        predicted = sim['mu_history'].cpu().numpy()

        ss_res = np.sum((observed - predicted)**2)
        ss_tot = np.sum((observed - np.mean(observed))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
        rmse = np.sqrt(np.mean((predicted - observed)**2))

        results[temp] = {'r2': r2, 'rmse': rmse, 'predicted': predicted, 'observed': observed}
        print(f"\n{temp}°C: R2 = {r2:.4f}, RMSE = {rmse:.4f}")

    overall_r2 = np.mean([r['r2'] for r in results.values()])
    overall_rmse = np.mean([r['rmse'] for r in results.values()])
    print(f"\nOverall: R2 = {overall_r2:.4f}, RMSE = {overall_rmse:.4f}")

    # ======================================================================
    # 7. GENERATE PLOTS
    # ======================================================================
    print("\n" + "-" * 80)
    print("Generating plots...")
    print("-" * 80)

    save_dir = Path(config['save_dir'])

    # Plot 1: Predicted vs Observed
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Interactive Galling Model V5: Predicted vs Observed COF', fontsize=14)

    for idx, (temp, result) in enumerate(sorted(results.items())):
        ax = axes[idx]
        cycles = np.arange(1, len(result['observed']) + 1)

        ax.plot(cycles, result['observed'], 'o-', color='#003f5c', label='Observed', markersize=3, alpha=0.7)
        ax.plot(cycles, result['predicted'], 's--', color='#ffa600', label='Predicted', markersize=3, alpha=0.7)
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('COF')
        ax.set_title(f'{temp}°C (R2 = {result["r2"]:.4f})')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'predicted_vs_observed.png', dpi=300)
    print(f"Saved: {save_dir / 'predicted_vs_observed.png'}")

    # Plot 2: Growth-Detachment Analysis
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    fig.suptitle('V5 Growth-Detachment Analysis', fontsize=14)

    colors = {165: '#003f5c', 167.5: '#bc5090', 170: '#ffa600'}

    for col_idx, temp in enumerate([165, 167.5, 170]):
        with torch.no_grad():
            sim = model.simulate_multiple_cycles(T=temp, n_cycles=150)

        mu = sim['mu_history'].cpu().numpy()
        mass = sim['M_history'].cpu().numpy()
        p_detach = sim['p_detach_history'].cpu().numpy()
        growth = sim['growth_events'].cpu().numpy()
        detach = sim['detach_events'].cpu().numpy()
        cycles = np.arange(1, 151)

        # Row 1: COF
        axes[0, col_idx].plot(cycles, mu, color=colors[temp], linewidth=1)
        axes[0, col_idx].set_title(f'{temp}°C - COF')
        axes[0, col_idx].set_ylabel('COF')
        axes[0, col_idx].grid(alpha=0.3)

        # Row 2: Mass
        axes[1, col_idx].plot(cycles, mass, color=colors[temp])
        axes[1, col_idx].set_title(f'{temp}°C - Transfer Mass')
        axes[1, col_idx].set_ylabel('Mass')
        axes[1, col_idx].grid(alpha=0.3)

        # Row 3: P(detach)
        axes[2, col_idx].plot(cycles, p_detach, color=colors[temp])
        axes[2, col_idx].set_title(f'{temp}°C - P(detachment)')
        axes[2, col_idx].set_ylabel('Probability')
        axes[2, col_idx].grid(alpha=0.3)

        # Row 4: Events
        g_cycles = cycles[growth > 0.5]
        d_cycles = cycles[detach > 0.5]
        axes[3, col_idx].scatter(g_cycles, np.ones_like(g_cycles),
                                c='green', s=10, alpha=0.7, label='Growth')
        axes[3, col_idx].scatter(d_cycles, np.zeros_like(d_cycles),
                                c='red', s=10, alpha=0.7, label='Detach')
        axes[3, col_idx].set_title(f'{temp}°C - Events')
        axes[3, col_idx].set_xlabel('Cycle')
        axes[3, col_idx].set_ylabel('Event')
        axes[3, col_idx].set_ylim(-0.3, 1.3)
        axes[3, col_idx].legend(fontsize=8)
        axes[3, col_idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'growth_detachment_analysis.png', dpi=300)
    print(f"Saved: {save_dir / 'growth_detachment_analysis.png'}")

    # ======================================================================
    # 8. SUMMARY
    # ======================================================================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY - V5 GROWTH + DETACHMENT")
    print("=" * 80)

    print("\nFinal physics parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:25s} = {value:.4f}")

    print("\nPerformance:")
    for temp in sorted(results.keys()):
        print(f"  {temp}°C: R2 = {results[temp]['r2']:.4f}, RMSE = {results[temp]['rmse']:.4f}")

    print(f"\nOverall: R2 = {overall_r2:.4f}, RMSE = {overall_rmse:.4f}")

    print("\n" + "=" * 80)
    print("V5 KEY INNOVATION:")
    print("=" * 80)
    print("  - Three-process model: growth + decay + detachment")
    print("  - Growth is stochastic with state-dependent rate (peaks at MID)")
    print("  - Detachment is mass+temperature dependent with variable severity")
    print("  - LOW state: growth rare, decay dominates -> stable")
    print("  - HIGH state: growth active, frequent detachments -> oscillations")
    print("  - Transition: uncertain competition -> chaotic bistability")
    print("=" * 80)


if __name__ == "__main__":
    main()

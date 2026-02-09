"""
Training Script for Interactive Galling Model V3 - Probabilistic Competition

V3 Key Features:
1. Discrete probabilistic competition instead of continuous noise
2. P(growth wins) depends on both physical rates AND state (β)
3. Natural stability ranking without explicit noise tuning

Oscillations emerge from stochastic competition outcomes, not random noise.

Usage:
    python scripts/train_interactive_galling_v3.py                    # Train from scratch
    python scripts/train_interactive_galling_v3.py --resume-from-best # Continue from best
    python scripts/train_interactive_galling_v3.py --validate-only    # Only validate

Author: Claude
Date: 2026-01-26
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
from src.models.interactive_galling_model_v3 import InteractiveGallingModelV3
from src.trainers.trainer_interactive_galling import InteractiveGallingTrainer, validate_model


def main():
    parser = argparse.ArgumentParser(description='Train Interactive Galling Model V3')
    parser.add_argument('--resume-from-best', action='store_true',
                        help='Resume training from best checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                        help='Skip training and only run validation')
    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING INTERACTIVE GALLING MODEL V3 - PROBABILISTIC COMPETITION")
    print("=" * 80)
    print("\nV3 Key Features:")
    print("  1. Discrete probabilistic competition (not continuous noise)")
    print("  2. P(growth) depends on physical rates + state (β)")
    print("  3. Natural stability: LOW=stable, HIGH=oscillating, MID=chaotic")
    print("=" * 80)

    # Configuration
    config = {
        'learning_rate': 0.005,
        'weight_decay': 1e-5,
        'n_epochs': 200,
        'n_likelihood_samples': 30,
        'kde_bandwidth': 0.08,
        'save_dir': 'results/interactive_galling_v3'
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
    # 2. CREATE V3 MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Creating Interactive Galling Model V3...")
    print("-" * 80)

    # V3 Physics parameters - Probabilistic competition
    physics_init = {
        # Growth Physics
        'k_adh': 0.05,
        'alpha_tau': 3.5,

        # Removal/Healing Physics
        'k_spall': 0.8,
        'E_spall': 0.5,

        # --- V3: Probabilistic Competition Parameters ---
        'beta_influence': 2.0,         # How nonlinear β affects probability
        'low_state_suppression': 0.1,  # P(growth) multiplier at β≈0
        'transition_boost': 1.5,       # Extra randomness at β≈0.5

        # Step sizes when growth/healing wins
        'growth_step': 0.3,
        'removal_step': 0.2,

        # Density Dynamics
        'M_sat': 5.0,

        # Transition Logic
        'rho_crit': 0.4,
        'beta_sharpness': 15.0,

        # Massive Detachment
        'prob_detach': 0.03,

        # Output Noise (measurement only)
        'output_noise': 0.02,
    }

    model = InteractiveGallingModelV3(
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
            save_dir='results/interactive_galling_v3/validation',
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
        p_growth = sim['p_growth_history'].cpu().numpy()
        outcomes = sim['outcome_history'].cpu().numpy()

        print(f"\n{temp}°C (100 cycles):")
        print(f"  COF: mean={mu.mean():.4f}, std={mu.std():.4f}")
        print(f"  β: mean={beta.mean():.4f}, final={beta[-1]:.4f}")
        print(f"  P(growth): mean={p_growth.mean():.4f}")
        print(f"  Growth wins: {outcomes.sum():.0f}/100 ({outcomes.mean()*100:.1f}%)")

    # ======================================================================
    # 4. TRAIN MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Training V3 model (likelihood-based)...")
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
    print("Validating trained V3 model...")
    print("-" * 80)

    validate_model(
        model=model,
        temp_data_dict=temp_data_dict,
        n_simulations=20,
        save_dir='results/interactive_galling_v3/validation',
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
    fig.suptitle('Interactive Galling Model V3: Predicted vs Observed COF', fontsize=14)

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

    # Plot 2: V3-specific - Probabilistic Competition Analysis
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    fig.suptitle('V3 Probabilistic Competition Analysis', fontsize=14)

    colors = {165: '#003f5c', 167.5: '#bc5090', 170: '#ffa600'}

    for col_idx, temp in enumerate([165, 167.5, 170]):
        with torch.no_grad():
            sim = model.simulate_multiple_cycles(T=temp, n_cycles=150)

        mu = sim['mu_history'].cpu().numpy()
        beta = sim['beta_history'].cpu().numpy()
        p_growth = sim['p_growth_history'].cpu().numpy()
        outcomes = sim['outcome_history'].cpu().numpy()
        cycles = np.arange(1, 151)

        # Row 1: COF
        axes[0, col_idx].plot(cycles, mu, color=colors[temp], linewidth=1)
        axes[0, col_idx].set_title(f'{temp}°C - COF')
        axes[0, col_idx].set_ylabel('COF')
        axes[0, col_idx].grid(alpha=0.3)

        # Row 2: Beta
        axes[1, col_idx].plot(cycles, beta, color=colors[temp])
        axes[1, col_idx].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[1, col_idx].set_title(f'{temp}°C - Regime (β)')
        axes[1, col_idx].set_ylabel('β')
        axes[1, col_idx].grid(alpha=0.3)

        # Row 3: Growth Probability
        axes[2, col_idx].plot(cycles, p_growth, color=colors[temp])
        axes[2, col_idx].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[2, col_idx].set_title(f'{temp}°C - P(growth wins)')
        axes[2, col_idx].set_ylabel('Probability')
        axes[2, col_idx].grid(alpha=0.3)

        # Row 4: Outcomes
        axes[3, col_idx].scatter(cycles, outcomes, c=outcomes, cmap='RdYlGn', s=10, alpha=0.7)
        axes[3, col_idx].set_title(f'{temp}°C - Outcomes')
        axes[3, col_idx].set_xlabel('Cycle')
        axes[3, col_idx].set_ylabel('Winner')
        axes[3, col_idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'probabilistic_competition.png', dpi=300)
    print(f"Saved: {save_dir / 'probabilistic_competition.png'}")

    # ======================================================================
    # 8. SUMMARY
    # ======================================================================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY - V3 PROBABILISTIC COMPETITION")
    print("=" * 80)

    print("\nFinal physics parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:25s} = {value:.4f}")

    print("\nPerformance:")
    for temp in sorted(results.keys()):
        print(f"  {temp}°C: R2 = {results[temp]['r2']:.4f}, RMSE = {results[temp]['rmse']:.4f}")

    print(f"\nOverall: R2 = {overall_r2:.4f}, RMSE = {overall_rmse:.4f}")

    print("\n" + "=" * 80)
    print("V3 KEY INNOVATION:")
    print("=" * 80)
    print("  Oscillations emerge from PROBABILISTIC COMPETITION:")
    print("  - LOW state (β≈0): Healing almost always wins → STABLE")
    print("  - HIGH state (β≈1): Balanced competition → OSCILLATING")
    print("  - MID state (β≈0.5): Maximum uncertainty → CHAOTIC")
    print("=" * 80)


if __name__ == "__main__":
    main()

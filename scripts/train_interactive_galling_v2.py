"""
Training Script for Interactive Galling Model V2 - Competition Dynamics

V2 Key Features:
1. Non-linear healing (M^n) - thick layers are unstable
2. Competition intensity amplification at transition
3. Constant output noise (more interpretable)

Oscillations emerge from PHYSICS, not state-dependent noise.

Usage:
    python scripts/train_interactive_galling_v2.py                    # Train from scratch
    python scripts/train_interactive_galling_v2.py --resume-from-best # Continue from best
    python scripts/train_interactive_galling_v2.py --validate-only    # Only validate

Author: Claude
Date: 2026-01-25
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.cycle_averaged_loader import CycleAveragedLoader
from src.models.interactive_galling_model_v2 import InteractiveGallingModelV2
from src.trainers.trainer_interactive_galling import InteractiveGallingTrainer, validate_model


def main():
    parser = argparse.ArgumentParser(description='Train Interactive Galling Model V2')
    parser.add_argument('--resume-from-best', action='store_true',
                        help='Resume training from best checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                        help='Skip training and only run validation')
    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING INTERACTIVE GALLING MODEL V2 - COMPETITION DYNAMICS")
    print("=" * 80)
    print("\nV2 Key Features:")
    print("  1. Non-linear healing: removal ∝ M^n (thick layers are unstable)")
    print("  2. Competition intensity: amplified dynamics at transition (β ≈ 0.5)")
    print("  3. Constant output noise (oscillations from physics, not noise)")
    print("=" * 80)

    # Configuration
    config = {
        'learning_rate': 0.005,
        'weight_decay': 1e-5,
        'n_epochs': 200,
        'n_likelihood_samples': 30,
        'kde_bandwidth': 0.08,
        'save_dir': 'results/interactive_galling_v2'  # NEW directory for V2
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
        print(f"    Range: [{data['mean_cof'].min():.4f}, {data['mean_cof'].max():.4f}]")

    # ======================================================================
    # 2. CREATE V2 MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Creating Interactive Galling Model V2...")
    print("-" * 80)

    # V2 Physics parameters - Competition dynamics
    physics_init = {
        # Growth Physics (same as V1)
        'k_adh': 0.05,       # Base adhesion rate
        'alpha_tau': 3.5,    # Shear sensitivity (feedback strength)

        # Removal/Healing Physics
        'k_spall': 0.8,      # Base spalling rate
        'E_spall': 0.5,      # Thermal activation for sticking

        # NEW V2: Non-linear Healing
        # n > 1: thick layers become unstable → creates oscillations at high M
        'healing_exponent': 1.5,

        # NEW V2: Competition Intensity
        # κ: amplification factor at transition
        # competition_intensity = 1 + κ·β(1-β), peaks at β = 0.5
        'competition_kappa': 2.0,

        # Density Dynamics
        'M_sat': 5.0,        # Saturation mass

        # Transition Logic
        'rho_crit': 0.4,     # Density threshold
        'beta_sharpness': 15.0,

        # Stochasticity
        'noise_lvl': 0.15,      # Process noise
        'prob_detach': 0.05,    # Detachment probability
        'output_noise': 0.05,   # CONSTANT output noise (V2 simplification)
    }

    model = InteractiveGallingModelV2(
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
            previous_epoch = checkpoint.get('epoch', 'N/A')
            print(f"  Previous training: epoch {previous_epoch}, log-likelihood = {previous_ll}")
        else:
            print(f"\nWARNING: Checkpoint not found at {checkpoint_path}")
            if args.validate_only:
                print("  ERROR: Cannot validate without a trained model!")
                return
            else:
                print("  Proceeding with random initialization")

    print("\nInitial physics parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:20s} = {value:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params}")

    # Skip to validation if requested
    if args.validate_only:
        print("\nSkipping training - proceeding to validation...")
        validate_model(
            model=model,
            temp_data_dict=temp_data_dict,
            n_simulations=20,
            save_dir='results/interactive_galling_v2/validation',
            device=device
        )
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE!")
        print("=" * 80)
        return

    # ======================================================================
    # 3. PRE-TRAINING CHECK
    # ======================================================================
    print("\n" + "-" * 80)
    print("Pre-training simulation check...")
    print("-" * 80)

    for temp in [165, 167.5, 170]:
        with torch.no_grad():
            sim = model.simulate_multiple_cycles(
                T=temp,
                n_cycles=100,
                M_init=0.0,
                rho_init=0.0,
                add_noise=True
            )
        mu = sim['mu_history'].cpu().numpy()
        beta = sim['beta_history'].cpu().numpy()
        competition = sim['competition_history'].cpu().numpy()

        print(f"\n{temp}°C (100 cycles):")
        print(f"  COF: mean={mu.mean():.4f}, std={mu.std():.4f}")
        print(f"  Beta: mean={beta.mean():.4f}, final={beta[-1]:.4f}")
        print(f"  Competition intensity: mean={competition.mean():.4f}, max={competition.max():.4f}")

    # ======================================================================
    # 4. TRAIN MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Training V2 model (likelihood-based)...")
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
    print("Validating trained V2 model...")
    print("-" * 80)

    validate_model(
        model=model,
        temp_data_dict=temp_data_dict,
        n_simulations=20,
        save_dir='results/interactive_galling_v2/validation',
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
            sim = model.simulate_multiple_cycles(
                T=temp,
                n_cycles=n_cycles,
                M_init=0.0,
                rho_init=0.0,
                add_noise=False
            )

        predicted = sim['mu_history'].cpu().numpy()

        ss_res = np.sum((observed - predicted)**2)
        ss_tot = np.sum((observed - np.mean(observed))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
        rmse = np.sqrt(np.mean((predicted - observed)**2))

        results[temp] = {
            'r2': r2,
            'rmse': rmse,
            'predicted': predicted,
            'observed': observed
        }

        print(f"\n{temp}°C:")
        print(f"  R2 = {r2:.4f}")
        print(f"  RMSE = {rmse:.4f}")

    overall_r2 = np.mean([r['r2'] for r in results.values()])
    overall_rmse = np.mean([r['rmse'] for r in results.values()])
    print(f"\nOverall: R2 = {overall_r2:.4f}, RMSE = {overall_rmse:.4f}")

    # ======================================================================
    # 7. GENERATE COMPARISON PLOTS
    # ======================================================================
    print("\n" + "-" * 80)
    print("Generating comparison plots...")
    print("-" * 80)

    save_dir = Path(config['save_dir'])

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Interactive Galling Model V2: Predicted vs Observed COF',
                 fontsize=14, fontweight='bold')

    for idx, (temp, result) in enumerate(sorted(results.items())):
        ax = axes[idx]

        observed = result['observed']
        predicted = result['predicted']
        cycles = np.arange(1, len(observed) + 1)

        ax.plot(cycles, observed, 'o-', color='#003f5c', label='Observed',
                markersize=3, alpha=0.7)
        ax.plot(cycles, predicted, 's--', color='#ffa600', label='Predicted',
                markersize=3, alpha=0.7)

        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Cycle-averaged COF')
        ax.set_title(f'{temp}°C (R2 = {result["r2"]:.4f}, RMSE = {result["rmse"]:.4f})')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'predicted_vs_observed.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'predicted_vs_observed.png'}")

    # ======================================================================
    # 8. V2-SPECIFIC: COMPETITION DYNAMICS PLOT
    # ======================================================================
    print("\nGenerating V2 competition dynamics plot...")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('V2 Competition Dynamics Analysis', fontsize=14, fontweight='bold')

    colors = {165: '#003f5c', 167.5: '#bc5090', 170: '#ffa600'}

    for col_idx, temp in enumerate([165, 167.5, 170]):
        with torch.no_grad():
            sim = model.simulate_multiple_cycles(T=temp, n_cycles=150)

        mu = sim['mu_history'].cpu().numpy()
        M = sim['M_history'].cpu().numpy()
        competition = sim['competition_history'].cpu().numpy()
        cycles = np.arange(1, 151)

        # Row 1: COF
        axes[0, col_idx].plot(cycles, mu, color=colors[temp])
        axes[0, col_idx].set_title(f'{temp}°C - COF')
        axes[0, col_idx].set_ylabel('COF')
        axes[0, col_idx].grid(alpha=0.3)

        # Row 2: Mass (M)
        axes[1, col_idx].plot(cycles, M, color=colors[temp])
        axes[1, col_idx].set_title(f'{temp}°C - Mass (M)')
        axes[1, col_idx].set_ylabel('M')
        axes[1, col_idx].grid(alpha=0.3)

        # Row 3: Competition Intensity
        axes[2, col_idx].plot(cycles, competition, color=colors[temp])
        axes[2, col_idx].set_title(f'{temp}°C - Competition Intensity')
        axes[2, col_idx].set_xlabel('Cycle')
        axes[2, col_idx].set_ylabel('Intensity')
        axes[2, col_idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'competition_dynamics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'competition_dynamics.png'}")

    # ======================================================================
    # 9. SUMMARY
    # ======================================================================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY - V2 COMPETITION DYNAMICS")
    print("=" * 80)

    print("\nFinal physics parameters:")
    final_params = model.get_physics_params()
    for name, value in final_params.items():
        print(f"  {name:20s} = {value:.4f}")

    print("\nPerformance:")
    for temp in sorted(results.keys()):
        r = results[temp]
        print(f"  {temp}°C: R2 = {r['r2']:.4f}, RMSE = {r['rmse']:.4f}")

    print(f"\nOverall: R2 = {overall_r2:.4f}, RMSE = {overall_rmse:.4f}")

    print("\n" + "=" * 80)
    print("V2 KEY FEATURES:")
    print("=" * 80)
    print("  1. Non-linear healing (M^n): thick layers are mechanically unstable")
    print("  2. Competition intensity: amplified dynamics at transition (beta~0.5)")
    print("  3. Constant output noise: oscillations emerge from PHYSICS")
    print("\nStability ranking (via physics, not noise):")
    print("  - Clean (beta~0): Slow dynamics -> stable")
    print("  - Galled (beta~1): M^n creates limit cycles -> oscillating")
    print("  - Transition (beta~0.5): Amplified competition -> chaotic")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {save_dir}/")
    print(f"Best model: {save_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()

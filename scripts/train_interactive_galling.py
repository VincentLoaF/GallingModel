"""
Training Script for Interactive Galling Model

This implements the new physics model based on:
- Xiao Yang's interactive friction model
- Galling density (ρ) as state variable
- Continuous formation/healing competition
- Probabilistic regime transitions

Usage:
    python scripts/train_interactive_galling.py                    # Train from scratch
    python scripts/train_interactive_galling.py --resume-from-best # Continue from best checkpoint
    python scripts/train_interactive_galling.py --validate-only    # Only validate saved model

Author: Claude
Date: 2026-01-15
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
from src.models.interactive_galling_model import InteractiveGallingModel
from src.trainers.trainer_interactive_galling import InteractiveGallingTrainer, validate_model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Interactive Galling Model')
    parser.add_argument('--resume-from-best', action='store_true',
                        help='Resume training from best saved checkpoint (fine-tuning)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Skip training and only run validation on saved model')
    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING INTERACTIVE GALLING MODEL")
    print("=" * 80)
    print("\nModel based on:")
    print("  - Xiao Yang's interactive friction formula: μ = (1-β)·μ_low + β·μ_high")
    print("  - Galling density (ρ) as state variable for surface coverage")
    print("  - Continuous competition: k_form vs k_heal")
    print("  - Probabilistic regime transitions with learnable thresholds")
    print("=" * 80)

    # Configuration
    config = {
        'learning_rate': 0.005,
        'weight_decay': 1e-5,
        'n_epochs': 200,
        'n_likelihood_samples': 30,
        'kde_bandwidth': 0.08,
        'save_dir': 'results/interactive_galling'
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if args.validate_only:
        print("\n📊 VALIDATION-ONLY MODE: Will load best model and run validation")
    elif args.resume_from_best:
        print("\n⚠️  FINE-TUNING MODE: Will load parameters from best checkpoint")

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
        print(f"    Mean COF: {data['mean_cof'].mean():.4f} ± {data['mean_cof'].std():.4f}")
        print(f"    Range: [{data['mean_cof'].min():.4f}, {data['mean_cof'].max():.4f}]")

    # ======================================================================
    # 2. CREATE MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Creating interactive galling model...")
    print("-" * 80)

    # Initial physics parameters for feedback loop model
    physics_init = {
        # Growth Physics (The Feedback Loop)
        'k_adh': 0.05,       # Base adhesion rate
        'alpha_tau': 3.5,    # Shear sensitivity (feedback strength)

        # Removal/Healing Physics (The Restoring Force)
        'k_spall': 0.8,      # Base spalling rate (healing)
        'E_spall': 0.5,      # Thermal activation energy for sticking

        # Density Dynamics
        'M_sat': 5.0,        # Saturation mass where rho approaches 1.0

        # Transition Logic
        'rho_crit': 0.4,     # Density threshold for regime shift
        'beta_sharpness': 15.0,  # How sharp the transition is

        # Stochasticity (Process Noise)
        'noise_lvl': 0.15,   # Random process noise for mass evolution
        'prob_detach': 0.05, # Probability of massive detachment event

        # State-Dependent Output Noise: σ(β) = σ_base + A·β + B·β(1-β)
        # Stability ranking: Clean (β=0) < Galled (β=1) < Transition (β=0.5)
        'sigma_base': 0.02,       # Baseline noise (clean state)
        'sigma_galled': 0.05,     # Additional noise at galled state (A)
        'sigma_transition': 0.15  # Transition instability peak (B), needs B > 2A
    }

    model = InteractiveGallingModel(
        T_ref=165.0,
        mu_low=0.15,
        mu_high=1.0,  # Changed from 1.05 to match model default
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

            previous_ll = checkpoint.get('best_log_likelihood', 'N/A')
            previous_epoch = checkpoint.get('epoch', 'N/A')
            print(f"  Previous training: epoch {previous_epoch}, log-likelihood = {previous_ll}")

            if args.validate_only:
                print("  ✓ Loaded trained model - will run validation only")
            else:
                print("  ✓ Loaded trained parameters - will fine-tune on all temperatures")
        else:
            print(f"\n⚠️  WARNING: Checkpoint not found at {checkpoint_path}")
            if args.validate_only:
                print("  ERROR: Cannot validate without a trained model!")
                return
            else:
                print("  Proceeding with random initialization instead")

    print("\nInitial physics parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:15s} = {value:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params}")

    # Skip to validation if requested
    if args.validate_only:
        print("\nSkipping training - proceeding directly to validation...")
        # Jump to validation section
        validate_model(
            model=model,
            temp_data_dict=temp_data_dict,
            n_simulations=20,
            save_dir='results/interactive_galling/validation',
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
        rho = sim['rho_history'].cpu().numpy()

        print(f"\n{temp}°C (100 cycles):")
        print(f"  COF: mean={mu.mean():.4f}, std={mu.std():.4f}, "
              f"range=[{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  β (regime): mean={beta.mean():.4f}, final={beta[-1]:.4f}")
        print(f"  ρ (density): mean={rho.mean():.4f}, final={rho[-1]:.4f}")

    # ======================================================================
    # 4. TRAIN MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Training model (likelihood-based)...")
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
    print("Validating trained model...")
    print("-" * 80)

    validate_model(
        model=model,
        temp_data_dict=temp_data_dict,
        n_simulations=20,
        save_dir='results/interactive_galling/validation',
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

        # Run deterministic simulation for metrics
        with torch.no_grad():
            sim = model.simulate_multiple_cycles(
                T=temp,
                n_cycles=n_cycles,
                M_init=0.0,
                rho_init=0.0,
                add_noise=False
            )

        predicted = sim['mu_history'].cpu().numpy()

        # Compute R^2
        ss_res = np.sum((observed - predicted)**2)
        ss_tot = np.sum((observed - np.mean(observed))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf

        # RMSE
        rmse = np.sqrt(np.mean((predicted - observed)**2))

        results[temp] = {
            'r2': r2,
            'rmse': rmse,
            'predicted': predicted,
            'observed': observed
        }

        print(f"\n{temp}°C:")
        print(f"  R² = {r2:.4f}")
        print(f"  RMSE = {rmse:.4f}")

    # Overall metrics
    overall_r2 = np.mean([r['r2'] for r in results.values()])
    overall_rmse = np.mean([r['rmse'] for r in results.values()])
    print(f"\nOverall: R² = {overall_r2:.4f}, RMSE = {overall_rmse:.4f}")

    # ======================================================================
    # 7. GENERATE COMPARISON PLOTS
    # ======================================================================
    print("\n" + "-" * 80)
    print("Generating comparison plots...")
    print("-" * 80)

    save_dir = Path(config['save_dir'])

    # Plot: Predicted vs Observed
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Interactive Galling Model: Predicted vs Observed COF',
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
        ax.set_title(f'{temp}°C (R² = {result["r2"]:.4f}, RMSE = {result["rmse"]:.4f})')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'predicted_vs_observed.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'predicted_vs_observed.png'}")

    # ======================================================================
    # 8. SUMMARY
    # ======================================================================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    print("\nFinal physics parameters:")
    final_params = model.get_physics_params()
    for name, value in final_params.items():
        print(f"  {name:15s} = {value:.4f}")

    print("\nPerformance:")
    for temp in sorted(results.keys()):
        r = results[temp]
        print(f"  {temp}°C: R² = {r['r2']:.4f}, RMSE = {r['rmse']:.4f}")

    print(f"\nOverall: R² = {overall_r2:.4f}, RMSE = {overall_rmse:.4f}")

    print("\n" + "=" * 80)
    print("KEY MODEL FEATURES:")
    print("=" * 80)
    print("  1. Galling density (ρ) as state variable - tracks surface coverage")
    print("  2. Continuous formation/healing competition:")
    print("     - Low T: k_heal > k_form → oscillating regime")
    print("     - High T: k_form > k_heal → permanent galling")
    print("  3. Probabilistic regime transitions with learnable thresholds")
    print("  4. Yang's interactive friction: μ = (1-β)·μ_low + β·μ_high")
    print("  5. Stochastic M_crit for spike variability")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    if args.resume_from_best:
        print("\n✓ Fine-tuning from best checkpoint completed successfully")
    elif not args.validate_only:
        print("\n✓ Training from scratch completed successfully")
    print(f"\nResults saved to: {save_dir}/")
    print(f"Best model: {save_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()

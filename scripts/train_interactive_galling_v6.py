"""
Training Script for Interactive Galling Model V6 - Empirical Transition Model

V6 uses direct Maximum Likelihood Estimation on observed cycle-to-cycle
COF transitions. No Monte Carlo sampling, no KDE, no simulation during
training. Just maximize P(observed Δμ | T, μ_current).

Usage:
    python scripts/train_interactive_galling_v6.py                    # Train from scratch
    python scripts/train_interactive_galling_v6.py --resume-from-best # Continue from best
    python scripts/train_interactive_galling_v6.py --validate-only    # Only validate

Date: 2026-02-02
"""

import sys
from pathlib import Path
import argparse
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.cycle_averaged_loader import CycleAveragedLoader
from src.models.interactive_galling_model_v6 import InteractiveGallingModelV6


def extract_transitions(temp_data_dict):
    """
    Extract (T, μ_current, Δμ) triples from training data.

    Returns:
        T_all: tensor of temperatures
        mu_all: tensor of current COF values
        delta_all: tensor of COF changes
    """
    T_list, mu_list, delta_list = [], [], []

    for temp, data in temp_data_dict.items():
        mu = data['mean_cof']
        n = len(mu)
        delta_mu = np.diff(mu)  # μ(n+1) - μ(n)
        mu_current = mu[:-1]

        T_list.append(np.full(n - 1, temp))
        mu_list.append(mu_current)
        delta_list.append(delta_mu)

    return (
        torch.tensor(np.concatenate(T_list), dtype=torch.float32),
        torch.tensor(np.concatenate(mu_list), dtype=torch.float32),
        torch.tensor(np.concatenate(delta_list), dtype=torch.float32),
    )


def train_model(model, T_all, mu_all, delta_all, config, device='cpu'):
    """
    Train V6 via direct MLE on observed transitions.

    Returns:
        history: dict with training metrics per epoch
    """
    T_all = T_all.to(device)
    mu_all = mu_all.to(device)
    delta_all = delta_all.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=30, factor=0.5
    )

    n_epochs = config['n_epochs']
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_ll = -float('inf')
    best_epoch = 0
    patience_counter = 0
    patience_limit = config.get('patience_limit', 50)

    history = {
        'train_log_likelihood': [],
        'per_temp_ll': [],
        'params': [],
    }

    # Pre-compute per-temperature masks for logging
    temp_masks = {}
    for temp in [165, 167.5, 170]:
        temp_masks[temp] = (T_all == temp)

    n_samples = len(T_all)
    print(f"\nTraining on {n_samples} transitions")
    print(f"  165°C: {temp_masks[165].sum().item()}")
    print(f"  167.5°C: {temp_masks[167.5].sum().item()}")
    print(f"  170°C: {temp_masks[170.0].sum().item()}")

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Compute log-likelihood
        log_probs = model.log_prob(T_all, mu_all, delta_all)
        total_ll = log_probs.sum()
        mean_ll = log_probs.mean()

        # Minimize negative log-likelihood
        loss = -mean_ll
        loss.backward()
        optimizer.step()
        scheduler.step(total_ll.item())

        # Per-temperature log-likelihood
        per_temp = {}
        for temp, mask in temp_masks.items():
            if mask.sum() > 0:
                per_temp[temp] = log_probs[mask].sum().item()

        # Track history
        history['train_log_likelihood'].append(total_ll.item())
        history['per_temp_ll'].append(per_temp)
        history['params'].append(model.get_physics_params())

        # Best model tracking
        if total_ll.item() > best_ll:
            best_ll = total_ll.item()
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_log_likelihood': best_ll,
                'per_temp_ll': per_temp,
            }, save_dir / 'best_model.pth')
        else:
            patience_counter += 1

        # Logging
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d} | LL={total_ll.item():8.2f} "
                  f"(best={best_ll:.2f} @{best_epoch}) | "
                  f"165={per_temp.get(165, 0):.1f} "
                  f"167.5={per_temp.get(167.5, 0):.1f} "
                  f"170={per_temp.get(170.0, 0):.1f} | "
                  f"lr={lr:.1e} | pat={patience_counter}")

        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience_limit} epochs)")
            break

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'log_likelihood': total_ll.item(),
    }, save_dir / 'final_model.pth')

    # Save history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best LL={best_ll:.2f} at epoch {best_epoch}")
    return history


def validate_model(model, temp_data_dict, n_simulations=20, save_dir='results/interactive_galling_v6/validation', device='cpu'):
    """
    Validate model by simulating trajectories and comparing to training data.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    colors_sim = '#ffa600'
    colors_obs = '#003f5c'

    for temp, data in sorted(temp_data_dict.items()):
        observed = data['mean_cof']
        n_cycles = data['n_cycles']

        # Simulate multiple trajectories
        simulations = []
        with torch.no_grad():
            for _ in range(n_simulations):
                sim = model.simulate(T=temp, n_cycles=n_cycles)
                simulations.append(sim['mu_history'].cpu().numpy())

        sim_array = np.array(simulations)
        cycles = np.arange(1, n_cycles + 1)

        # --- Multi-panel validation plot ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'V6 Validation - {temp}°C', fontsize=14)

        # Panel 1: Trajectories overlay
        ax = axes[0, 0]
        for i, sim in enumerate(simulations):
            ax.plot(cycles, sim, color=colors_sim, alpha=0.15, linewidth=0.8,
                    label='Simulated' if i == 0 else None)
        ax.plot(cycles, observed, color=colors_obs, linewidth=1.5, label='Observed')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('COF')
        ax.set_title(f'{n_simulations} Simulated Trajectories vs Observed')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 2: Distribution comparison (histogram)
        ax = axes[0, 1]
        ax.hist(observed, bins=30, density=True, alpha=0.7, color=colors_obs, label='Observed')
        all_sim = sim_array.flatten()
        ax.hist(all_sim, bins=30, density=True, alpha=0.5, color=colors_sim, label='Simulated')
        ax.set_xlabel('COF')
        ax.set_ylabel('Density')
        ax.set_title('COF Distribution')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 3: Mean ± std envelope
        ax = axes[1, 0]
        sim_mean = sim_array.mean(axis=0)
        sim_std = sim_array.std(axis=0)
        ax.fill_between(cycles, sim_mean - sim_std, sim_mean + sim_std,
                        alpha=0.3, color=colors_sim, label='Sim mean ± std')
        ax.plot(cycles, sim_mean, color=colors_sim, linewidth=1.5)
        ax.plot(cycles, observed, color=colors_obs, linewidth=1.5, label='Observed')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('COF')
        ax.set_title('Simulation Envelope')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 4: Single best trajectory
        ax = axes[1, 1]
        # Pick trajectory with closest mean to observed
        obs_mean = observed.mean()
        best_idx = np.argmin([abs(s.mean() - obs_mean) for s in simulations])
        ax.plot(cycles, observed, color=colors_obs, linewidth=1.5, label='Observed')
        ax.plot(cycles, simulations[best_idx], color=colors_sim, linewidth=1.0,
                linestyle='--', label='Best simulation')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('COF')
        ax.set_title('Closest Trajectory Match')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f'validation_{temp}C.png', dpi=200)
        plt.close()
        print(f"  Saved: {save_dir / f'validation_{temp}C.png'}")

        # Print stats
        print(f"\n  {temp}°C validation:")
        print(f"    Observed: mean={observed.mean():.4f}, std={observed.std():.4f}")
        print(f"    Simulated: mean={all_sim.mean():.4f}, std={all_sim.std():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Interactive Galling Model V6')
    parser.add_argument('--resume-from-best', action='store_true',
                        help='Resume training from best checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                        help='Skip training and only run validation')
    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING INTERACTIVE GALLING MODEL V6 - EMPIRICAL TRANSITIONS")
    print("=" * 80)
    print("\nV6 Key Change:")
    print("  Direct MLE on observed transitions P(Δμ | T, μ)")
    print("  2-component Gaussian mixture: 'stay' + 'jump'")
    print("  No hidden state (mass, density). No MC sampling. No KDE.")
    print("  17 learnable parameters in analytic functions.")
    print("=" * 80)

    config = {
        'learning_rate': 0.01,
        'weight_decay': 1e-5,
        'n_epochs': 5000,
        'patience_limit': 500,
        'save_dir': 'results/interactive_galling_v6',
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

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
    # 2. EXTRACT TRANSITIONS
    # ======================================================================
    print("\n" + "-" * 80)
    print("Extracting transitions...")
    print("-" * 80)

    T_all, mu_all, delta_all = extract_transitions(temp_data_dict)
    print(f"\nTotal transitions: {len(T_all)}")

    # Print transition stats
    for temp in [165, 167.5, 170]:
        mask = (T_all == temp)
        d = delta_all[mask]
        m = mu_all[mask]
        print(f"\n  {temp}°C: {mask.sum().item()} transitions")
        print(f"    Δμ: mean={d.mean():.4f}, std={d.std():.4f}, "
              f"range=[{d.min():.4f}, {d.max():.4f}]")
        print(f"    μ: mean={m.mean():.4f}, range=[{m.min():.4f}, {m.max():.4f}]")

    # ======================================================================
    # 3. CREATE MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Creating V6 model...")
    print("-" * 80)

    model = InteractiveGallingModelV6(device=device)
    model = model.to(device)

    # Load checkpoint if requested
    if args.validate_only or args.resume_from_best:
        checkpoint_path = Path(config['save_dir']) / 'best_model.pth'
        if checkpoint_path.exists():
            print(f"\nLoading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            previous_ll = checkpoint.get('best_log_likelihood', 'N/A')
            print(f"  Previous log-likelihood: {previous_ll}")
        else:
            print(f"\nWARNING: Checkpoint not found at {checkpoint_path}")
            if args.validate_only:
                return

    print("\nInitial parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:15s} = {value:.4f}")

    # Skip to validation if requested
    if args.validate_only:
        print("\nSkipping training - proceeding to validation...")
        validate_model(model, temp_data_dict, n_simulations=20,
                      save_dir=config['save_dir'] + '/validation', device=device)
        return

    # ======================================================================
    # 4. PRE-TRAINING CHECK
    # ======================================================================
    print("\n" + "-" * 80)
    print("Pre-training: initial log-likelihood...")
    print("-" * 80)

    with torch.no_grad():
        T_dev = T_all.to(device)
        mu_dev = mu_all.to(device)
        delta_dev = delta_all.to(device)
        init_ll = model.log_prob(T_dev, mu_dev, delta_dev).sum().item()
    print(f"  Initial total LL: {init_ll:.2f}")

    print("\nPre-training simulation (with initial params):")
    for temp in [165, 167.5, 170]:
        with torch.no_grad():
            sim = model.simulate(T=temp, n_cycles=100)
        mu = sim['mu_history'].cpu().numpy()
        comp = sim['component_history'].cpu().numpy()
        print(f"  {temp}°C: mean={mu.mean():.4f}, std={mu.std():.4f}, "
              f"jumps={comp.sum():.0f}/100")

    # ======================================================================
    # 5. TRAIN
    # ======================================================================
    print("\n" + "-" * 80)
    print("Training V6 model (direct MLE)...")
    print("-" * 80)

    history = train_model(model, T_all, mu_all, delta_all, config, device=device)

    # Load best model for validation
    checkpoint = torch.load(Path(config['save_dir']) / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # ======================================================================
    # 6. VALIDATE
    # ======================================================================
    print("\n" + "-" * 80)
    print("Validating trained V6 model...")
    print("-" * 80)

    validate_model(model, temp_data_dict, n_simulations=20,
                  save_dir=config['save_dir'] + '/validation', device=device)

    # ======================================================================
    # 7. GENERATE PLOTS
    # ======================================================================
    print("\n" + "-" * 80)
    print("Generating plots...")
    print("-" * 80)

    save_dir = Path(config['save_dir'])

    # Plot 1: Training loss curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epochs = range(len(history['train_log_likelihood']))
    ax.plot(epochs, history['train_log_likelihood'], 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Log-Likelihood')
    ax.set_title('V6 Training Progress')
    ax.grid(alpha=0.3)
    best_epoch = np.argmax(history['train_log_likelihood'])
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
               label=f'Best epoch {best_epoch}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curve.png', dpi=200)
    print(f"  Saved: {save_dir / 'training_curve.png'}")

    # Plot 2: Predicted vs Observed overlay (3 temps)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('V6: Simulated vs Observed COF', fontsize=14)

    for idx, (temp, data) in enumerate(sorted(temp_data_dict.items())):
        ax = axes[idx]
        observed = data['mean_cof']
        n_cycles = data['n_cycles']
        cycles = np.arange(1, n_cycles + 1)

        # Multiple simulations
        with torch.no_grad():
            for i in range(10):
                sim = model.simulate(T=temp, n_cycles=n_cycles)
                mu_sim = sim['mu_history'].cpu().numpy()
                ax.plot(cycles, mu_sim, color='#ffa600', alpha=0.2, linewidth=0.8,
                        label='Simulated' if i == 0 else None)

        ax.plot(cycles, observed, 'o-', color='#003f5c', label='Observed',
                markersize=2, linewidth=1)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('COF')
        ax.set_title(f'{temp}°C')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'predicted_vs_observed.png', dpi=200)
    print(f"  Saved: {save_dir / 'predicted_vs_observed.png'}")

    # Plot 3: Learned transition landscape
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('V6: Learned Transition Parameters', fontsize=14)

    mu_range = torch.linspace(0.1, 1.3, 100, device=device)

    for col_idx, temp in enumerate([165, 167.5, 170]):
        T_t = torch.tensor(temp, dtype=torch.float32, device=device)
        with torch.no_grad():
            pi, d1, s1, d2, s2 = model.transition_params(T_t, mu_range)
            pi = pi.cpu().numpy()
            d1 = d1.cpu().numpy()
            s1 = s1.cpu().numpy()
            d2 = d2.cpu().numpy()
            s2 = s2.cpu().numpy()

        mu_np = mu_range.cpu().numpy()

        # Row 0: mixing weight and drifts
        ax = axes[0, col_idx]
        ax.plot(mu_np, pi, 'b-', linewidth=2, label='π (P stay)')
        ax.set_title(f'{temp}°C')
        ax.set_ylabel('P(stay)')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Current μ')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Row 1: component parameters
        ax = axes[1, col_idx]
        ax.plot(mu_np, d1, 'g-', linewidth=1.5, label='d₁ (stay drift)')
        ax.fill_between(mu_np, d1 - s1, d1 + s1, alpha=0.2, color='green')
        ax.plot(mu_np, d2, 'r-', linewidth=1.5, label='d₂ (jump drift)')
        ax.fill_between(mu_np, d2 - s2, d2 + s2, alpha=0.2, color='red')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Current μ')
        ax.set_ylabel('Δμ')
        ax.set_title(f'{temp}°C - Components (mean ± σ)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'transition_landscape.png', dpi=200)
    print(f"  Saved: {save_dir / 'transition_landscape.png'}")

    # ======================================================================
    # 8. SUMMARY
    # ======================================================================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY - V6 EMPIRICAL TRANSITIONS")
    print("=" * 80)

    print("\nFinal parameters:")
    for name, value in model.get_physics_params().items():
        print(f"  {name:15s} = {value:.4f}")

    best_ll = max(history['train_log_likelihood'])
    best_epoch = np.argmax(history['train_log_likelihood'])
    print(f"\nBest total LL: {best_ll:.2f} at epoch {best_epoch}")

    best_per_temp = history['per_temp_ll'][best_epoch]
    print("\nPer-temperature LL:")
    for temp in sorted(best_per_temp.keys(), key=float):
        print(f"  {temp}°C: {best_per_temp[temp]:.2f}")

    print("\n" + "=" * 80)
    print("V6 KEY INNOVATION:")
    print("=" * 80)
    print("  - Direct MLE on observed transitions (no simulation during training)")
    print("  - 2-component Gaussian mixture: 'stay' + 'jump'")
    print("  - No hidden state: P(Δμ | T, μ) is everything")
    print("  - 17 learnable analytic parameters")
    print("  - Training uses exact gradients (no MC noise)")
    print("=" * 80)


if __name__ == "__main__":
    main()

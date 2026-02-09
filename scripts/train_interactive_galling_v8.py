"""
Training Script for Interactive Galling Model V8 - Double-Well Potential

V8 replaces V7's linear parameter functions with physics derived from a
quartic double-well potential. The same 2-component Gaussian mixture likelihood
is used, but all 5 mixture parameters (pi, d1, sigma1, d2, sigma2) are now
derived from the potential landscape rather than fitted as linear functions.

Architecture: 13 learnable parameters (vs V7's 17 linear coefficients).

Usage:
    python scripts/train_interactive_galling_v8.py                    # Train from scratch
    python scripts/train_interactive_galling_v8.py --resume-from-best # Continue from best
    python scripts/train_interactive_galling_v8.py --validate-only    # Only validate

Date: 2026-02-09
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
from src.models.interactive_galling_model_v8 import InteractiveGallingModelV8


def extract_transitions(temp_data_dict, temp_mapping=None):
    """
    Extract (T, mu_current, delta_mu) triples from training data.

    Args:
        temp_data_dict: dict mapping temperature -> data
        temp_mapping: optional dict mapping actual temp -> model temp
                      e.g., {25: 160} to treat 25 C as 160 C

    Returns:
        T_all: tensor of temperatures (after mapping)
        mu_all: tensor of current COF values
        delta_all: tensor of COF changes
    """
    if temp_mapping is None:
        temp_mapping = {}

    T_list, mu_list, delta_list = [], [], []

    for temp, data in temp_data_dict.items():
        mu = data['mean_cof']
        n = len(mu)
        delta_mu = np.diff(mu)  # mu(n+1) - mu(n)
        mu_current = mu[:-1]

        # Apply temperature mapping (e.g., 25 C -> 160 C)
        model_temp = temp_mapping.get(temp, temp)

        T_list.append(np.full(n - 1, model_temp))
        mu_list.append(mu_current)
        delta_list.append(delta_mu)

    return (
        torch.tensor(np.concatenate(T_list), dtype=torch.float32),
        torch.tensor(np.concatenate(mu_list), dtype=torch.float32),
        torch.tensor(np.concatenate(delta_list), dtype=torch.float32),
    )


def train_model(model, T_all, mu_all, delta_all, config, temp_list, device='cpu'):
    """
    Train V8 via direct MLE on observed transitions.

    Includes gradient clipping (max_norm=5.0) and NaN checks.

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
    for temp in temp_list:
        temp_masks[temp] = (T_all == temp)

    n_samples = len(T_all)
    print(f"\nTraining on {n_samples} transitions")
    print(f"  160 C: {temp_masks[160].sum().item()}")
    print(f"  165 C: {temp_masks[165].sum().item()}")
    print(f"  167.5 C: {temp_masks[167.5].sum().item()}")
    print(f"  170 C: {temp_masks[170.0].sum().item()}")

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Compute log-likelihood
        log_probs = model.log_prob(T_all, mu_all, delta_all)
        total_ll = log_probs.sum()
        mean_ll = log_probs.mean()

        # NaN check
        if torch.isnan(total_ll) or torch.isinf(total_ll):
            print(f"  WARNING: NaN/Inf detected at epoch {epoch}, skipping update")
            continue

        # Minimize negative log-likelihood
        loss = -mean_ll
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

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
                  f"160={per_temp.get(160, 0):.1f} "
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


def validate_model(model, temp_data_dict, n_simulations=20,
                   save_dir='results/interactive_galling_v8/validation',
                   device='cpu', temp_mapping=None):
    """
    Validate model by simulating trajectories and comparing to training data.

    Args:
        model: trained V8 model
        temp_data_dict: dict mapping actual temperature -> data
        n_simulations: number of simulations per temperature
        save_dir: directory to save validation plots
        device: torch device
        temp_mapping: dict mapping actual temp -> model temp for simulation
    """
    if temp_mapping is None:
        temp_mapping = {}

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    colors_sim = '#ffa600'
    colors_obs = '#003f5c'

    for temp, data in sorted(temp_data_dict.items()):
        observed = data['mean_cof']
        n_cycles = data['n_cycles']

        # Use mapped temperature for simulation
        sim_temp = temp_mapping.get(temp, temp)

        # Simulate multiple trajectories
        simulations = []
        with torch.no_grad():
            for _ in range(n_simulations):
                sim = model.simulate(T=sim_temp, n_cycles=n_cycles)
                simulations.append(sim['mu_history'].cpu().numpy())

        sim_array = np.array(simulations)
        cycles = np.arange(1, n_cycles + 1)

        # --- Multi-panel validation plot ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        title_note = f" (simulated as {sim_temp} C)" if temp in temp_mapping else ""
        fig.suptitle(f'V8 Validation - {temp} C{title_note}', fontsize=14)

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

        # Panel 3: Mean +/- std envelope
        ax = axes[1, 0]
        sim_mean = sim_array.mean(axis=0)
        sim_std = sim_array.std(axis=0)
        ax.fill_between(cycles, sim_mean - sim_std, sim_mean + sim_std,
                        alpha=0.3, color=colors_sim, label='Sim mean +/- std')
        ax.plot(cycles, sim_mean, color=colors_sim, linewidth=1.5)
        ax.plot(cycles, observed, color=colors_obs, linewidth=1.5, label='Observed')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('COF')
        ax.set_title('Simulation Envelope')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 4: Single best trajectory
        ax = axes[1, 1]
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
        print(f"\n  {temp} C validation:")
        print(f"    Observed: mean={observed.mean():.4f}, std={observed.std():.4f}")
        print(f"    Simulated: mean={all_sim.mean():.4f}, std={all_sim.std():.4f}")


def plot_potential_landscape(model, save_dir, device='cpu'):
    """
    V8-specific: 2x4 grid showing V(mu) and F(mu) at 160/165/167.5/170 C
    with well position markers.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    temps = [160, 165, 167.5, 170]
    colors = {160: '#2ca02c', 165: '#003f5c', 167.5: '#bc5090', 170: '#ffa600'}

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle('V8: Double-Well Potential Landscape', fontsize=14)

    model.eval()
    with torch.no_grad():
        mu_L = model.params['mu_L'].item()
        mu_H = model.params['mu_H'].item()
        mu_mid = (mu_L + mu_H) / 2.0

        for col_idx, temp in enumerate(temps):
            mu_range, V_values, force_values = model.get_potential_landscape(temp, n_points=300)
            mu_np = mu_range.cpu().numpy()
            V_np = V_values.cpu().numpy()
            F_np = force_values.cpu().numpy()
            color = colors[temp]

            # Row 0: Potential V(mu)
            ax = axes[0, col_idx]
            ax.plot(mu_np, V_np, color=color, linewidth=2)
            # Mark well positions
            ax.axvline(mu_L, color='green', linestyle=':', alpha=0.6, label=f'mu_L={mu_L:.2f}')
            ax.axvline(mu_H, color='red', linestyle=':', alpha=0.6, label=f'mu_H={mu_H:.2f}')
            ax.axvline(mu_mid, color='gray', linestyle='--', alpha=0.4, label=f'mu_mid={mu_mid:.2f}')
            ax.set_title(f'{temp} C - V(mu)')
            ax.set_ylabel('V(mu)')
            ax.set_xlabel('mu')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            # Row 1: Force F(mu) = -dV/dmu
            ax = axes[1, col_idx]
            ax.plot(mu_np, F_np, color=color, linewidth=2)
            ax.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax.axvline(mu_L, color='green', linestyle=':', alpha=0.6)
            ax.axvline(mu_H, color='red', linestyle=':', alpha=0.6)
            ax.set_title(f'{temp} C - F(mu) = -dV/dmu')
            ax.set_ylabel('-dV/dmu')
            ax.set_xlabel('mu')
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'potential_landscape.png', dpi=200)
    plt.close()
    print(f"  Saved: {save_dir / 'potential_landscape.png'}")


def plot_transition_params(model, save_dir, device='cpu'):
    """
    2x4 grid showing pi(mu) and component drifts +/- sigma at each temperature.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    temps = [160, 165, 167.5, 170]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle('V8: Learned Transition Parameters', fontsize=14)

    mu_range = torch.linspace(0.1, 1.3, 100, device=device)

    model.eval()
    with torch.no_grad():
        for col_idx, temp in enumerate(temps):
            T_t = torch.tensor(temp, dtype=torch.float32, device=device)
            pi, d1, s1, d2, s2 = model.transition_params(T_t, mu_range)
            pi = pi.cpu().numpy()
            d1 = d1.cpu().numpy()
            s1 = s1.cpu().numpy()
            d2 = d2.cpu().numpy()
            s2 = s2.cpu().numpy()

            mu_np = mu_range.cpu().numpy()

            # Row 0: mixing weight pi(mu)
            ax = axes[0, col_idx]
            ax.plot(mu_np, pi, 'b-', linewidth=2, label='pi (P stay)')
            ax.set_title(f'{temp} C')
            ax.set_ylabel('P(stay)')
            ax.set_ylim(0, 1)
            ax.set_xlabel('Current mu')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # Row 1: component parameters (drift +/- sigma)
            ax = axes[1, col_idx]
            ax.plot(mu_np, d1, 'g-', linewidth=1.5, label='d1 (stay drift)')
            ax.fill_between(mu_np, d1 - s1, d1 + s1, alpha=0.2, color='green')
            ax.plot(mu_np, d2, 'r-', linewidth=1.5, label='d2 (jump drift)')
            ax.fill_between(mu_np, d2 - s2, d2 + s2, alpha=0.2, color='red')
            ax.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Current mu')
            ax.set_ylabel('delta_mu')
            ax.set_title(f'{temp} C - Components (mean +/- sigma)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'transition_params.png', dpi=200)
    plt.close()
    print(f"  Saved: {save_dir / 'transition_params.png'}")


def plot_barrier_analysis(model, save_dir, device='cpu'):
    """
    2-panel barrier analysis: barrier height dV(mu) and escape rate k(mu)
    at each temperature.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    temps = [160, 165, 167.5, 170]
    colors = {160: '#2ca02c', 165: '#003f5c', 167.5: '#bc5090', 170: '#ffa600'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('V8: Barrier Analysis', fontsize=14)

    mu_range = torch.linspace(0.1, 1.3, 200, device=device)

    model.eval()
    with torch.no_grad():
        for temp in temps:
            T_t = torch.tensor(temp, dtype=torch.float32, device=device)
            color = colors[temp]

            # Barrier height
            dV = model.barrier_height(mu_range, T_t)
            dV_np = dV.cpu().numpy()

            # Noise intensity for escape rate
            D = model.noise_intensity(T_t)
            omega0 = torch.abs(model.params['omega0'])

            # Escape rate k = omega0 * exp(-dV / D)
            exp_arg = torch.clamp(-dV / (D + 1e-8), -20.0, 20.0)
            k = omega0 * torch.exp(exp_arg)
            k_np = k.cpu().numpy()

            mu_np = mu_range.cpu().numpy()

            # Panel 1: Barrier height
            axes[0].plot(mu_np, dV_np, color=color, linewidth=1.5, label=f'{temp} C')

            # Panel 2: Escape rate
            axes[1].plot(mu_np, k_np, color=color, linewidth=1.5, label=f'{temp} C')

    axes[0].set_xlabel('mu')
    axes[0].set_ylabel('dV (barrier height)')
    axes[0].set_title('Barrier Height dV(mu)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('mu')
    axes[1].set_ylabel('k (escape rate)')
    axes[1].set_title('Escape Rate k(mu)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'barrier_analysis.png', dpi=200)
    plt.close()
    print(f"  Saved: {save_dir / 'barrier_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description='Train Interactive Galling Model V8')
    parser.add_argument('--resume-from-best', action='store_true',
                        help='Resume training from best checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                        help='Skip training and only run validation')
    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING INTERACTIVE GALLING MODEL V8 - DOUBLE-WELL POTENTIAL")
    print("=" * 80)
    print("\nV8 Key Changes from V7:")
    print("  Quartic double-well potential V(mu,T) replaces linear parameter functions")
    print("  All 5 mixture params (pi, d1, sigma1, d2, sigma2) derived from potential")
    print("  13 learnable parameters (vs V7's 17 linear coefficients)")
    print("  Linearized barrier: h(T) = h0 * exp(-alpha_h * dT)")
    print("  Training data: 25 C -> 160 C, 165 C, 167.5 C, 170 C")
    print("=" * 80)

    # Temperature mapping: 25 C treated as 160 C (both are "no galling" baseline)
    TEMP_MAPPING = {25: 160}

    config = {
        'learning_rate': 0.005,
        'weight_decay': 1e-5,
        'n_epochs': 8000,
        'patience_limit': 1000,
        'save_dir': 'results/interactive_galling_v8',
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
    # Load all 4 temperatures including 25 C
    temp_data_dict = loader.load_all_temperatures([25, 165, 167.5, 170])

    print("\nDataset summary:")
    for temp, data in sorted(temp_data_dict.items()):
        mapped = TEMP_MAPPING.get(temp, temp)
        mapping_note = f" -> mapped to {mapped} C" if temp in TEMP_MAPPING else ""
        print(f"  {temp} C{mapping_note}: {data['n_cycles']} cycles")
        print(f"    Mean COF: {data['mean_cof'].mean():.4f} +/- {data['mean_cof'].std():.4f}")

    # ======================================================================
    # 2. EXTRACT TRANSITIONS
    # ======================================================================
    print("\n" + "-" * 80)
    print("Extracting transitions (with 25 C -> 160 C mapping)...")
    print("-" * 80)

    T_all, mu_all, delta_all = extract_transitions(temp_data_dict, temp_mapping=TEMP_MAPPING)
    print(f"\nTotal transitions: {len(T_all)}")

    # Model temperatures after mapping
    model_temps = [160, 165, 167.5, 170]

    # Print transition stats
    for temp in model_temps:
        mask = (T_all == temp)
        if mask.sum() == 0:
            continue
        d = delta_all[mask]
        m = mu_all[mask]
        source_note = " (from 25 C data)" if temp == 160 else ""
        print(f"\n  {temp} C{source_note}: {mask.sum().item()} transitions")
        print(f"    delta_mu: mean={d.mean():.4f}, std={d.std():.4f}, "
              f"range=[{d.min():.4f}, {d.max():.4f}]")
        print(f"    mu: mean={m.mean():.4f}, range=[{m.min():.4f}, {m.max():.4f}]")

    # ======================================================================
    # 3. CREATE MODEL
    # ======================================================================
    print("\n" + "-" * 80)
    print("Creating V8 model...")
    print("-" * 80)

    model = InteractiveGallingModelV8(device=device)
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
        print(f"  {name:20s} = {value:.6f}")

    # Skip to validation if requested
    if args.validate_only:
        print("\nSkipping training - proceeding to validation...")
        validate_model(model, temp_data_dict, n_simulations=20,
                      save_dir=config['save_dir'] + '/validation', device=device,
                      temp_mapping=TEMP_MAPPING)
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
    for temp in [160, 165, 167.5, 170]:
        with torch.no_grad():
            sim = model.simulate(T=temp, n_cycles=100)
        mu = sim['mu_history'].cpu().numpy()
        comp = sim['component_history'].cpu().numpy()
        print(f"  {temp} C: mean={mu.mean():.4f}, std={mu.std():.4f}, "
              f"jumps={comp.sum():.0f}/100")

    # ======================================================================
    # 5. TRAIN
    # ======================================================================
    print("\n" + "-" * 80)
    print("Training V8 model (direct MLE with gradient clipping)...")
    print("-" * 80)

    history = train_model(model, T_all, mu_all, delta_all, config, model_temps, device=device)

    # Load best model for validation
    checkpoint = torch.load(Path(config['save_dir']) / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # ======================================================================
    # 6. VALIDATE
    # ======================================================================
    print("\n" + "-" * 80)
    print("Validating trained V8 model...")
    print("-" * 80)

    validate_model(model, temp_data_dict, n_simulations=20,
                  save_dir=config['save_dir'] + '/validation', device=device,
                  temp_mapping=TEMP_MAPPING)

    # ======================================================================
    # 7. GENERATE PLOTS
    # ======================================================================
    print("\n" + "-" * 80)
    print("Generating plots...")
    print("-" * 80)

    save_dir = Path(config['save_dir'])

    # Plot 1: Training loss curve (with best epoch marker)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epochs = range(len(history['train_log_likelihood']))
    ax.plot(epochs, history['train_log_likelihood'], 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Log-Likelihood')
    ax.set_title('V8 Training Progress')
    ax.grid(alpha=0.3)
    best_epoch = np.argmax(history['train_log_likelihood'])
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
               label=f'Best epoch {best_epoch}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curve.png', dpi=200)
    plt.close()
    print(f"  Saved: {save_dir / 'training_curve.png'}")

    # Plot 2: Predicted vs Observed overlay (4 temps including 25 C)
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle('V8: Simulated vs Observed COF', fontsize=14)

    for idx, (temp, data) in enumerate(sorted(temp_data_dict.items())):
        # Use mapped temperature for simulation
        sim_temp = TEMP_MAPPING.get(temp, temp)
        ax = axes[idx]
        observed = data['mean_cof']
        n_cycles = data['n_cycles']
        cycles = np.arange(1, n_cycles + 1)

        # Multiple simulations (use mapped temperature)
        with torch.no_grad():
            for i in range(10):
                sim = model.simulate(T=sim_temp, n_cycles=n_cycles)
                mu_sim = sim['mu_history'].cpu().numpy()
                ax.plot(cycles, mu_sim, color='#ffa600', alpha=0.2, linewidth=0.8,
                        label='Simulated' if i == 0 else None)

        ax.plot(cycles, observed, 'o-', color='#003f5c', label='Observed',
                markersize=2, linewidth=1)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('COF')
        title_note = f" (simulated as {sim_temp} C)" if temp in TEMP_MAPPING else ""
        ax.set_title(f'{temp} C{title_note}')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'predicted_vs_observed.png', dpi=200)
    plt.close()
    print(f"  Saved: {save_dir / 'predicted_vs_observed.png'}")

    # Plot 3: Potential landscape (V8-specific)
    plot_potential_landscape(model, save_dir, device=device)

    # Plot 4: Transition parameters
    plot_transition_params(model, save_dir, device=device)

    # Plot 5: Barrier analysis
    plot_barrier_analysis(model, save_dir, device=device)

    # ======================================================================
    # 8. SUMMARY
    # ======================================================================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY - V8 DOUBLE-WELL POTENTIAL")
    print("=" * 80)

    print("\nAll parameters (6 decimal places):")
    params = model.get_physics_params()
    for name, value in params.items():
        print(f"  {name:20s} = {value:.6f}")

    best_ll = max(history['train_log_likelihood'])
    best_epoch = np.argmax(history['train_log_likelihood'])
    print(f"\nBest total LL: {best_ll:.2f} at epoch {best_epoch}")

    best_per_temp = history['per_temp_ll'][best_epoch]
    print("\nPer-temperature LL:")
    for temp in sorted(best_per_temp.keys(), key=float):
        print(f"  {temp} C: {best_per_temp[temp]:.2f}")

    # Physics interpretation
    print("\nPhysics Interpretation:")
    mu_L = params['mu_L']
    mu_H = params['mu_H']
    h0 = abs(params['h0'])
    g0 = params.get('g0', 0)
    g_T = params.get('g_T', 0)
    D0 = abs(params['D0'])
    D_T = params['D_T']
    omega0 = abs(params['omega0'])
    tau = abs(params['tau'])

    print(f"  Well positions: mu_L = {mu_L:.4f}, mu_H = {mu_H:.4f}")
    print(f"  Barrier height: h0 = {h0:.6f}")
    print(f"  Tilt at T_ref: g0 = {g0:.6f}")
    print(f"  Tilt sensitivity: g_T = {g_T:.6f} /°C")
    for dT_val in [0, 5, 7.5, 10]:
        g_val = g0 + g_T * dT_val
        T_val = 160 + dT_val
        print(f"    g({T_val}°C) = {g_val:.6f} ({'favors galled' if g_val > 0 else 'favors clean'})")
    print(f"  Noise at T_ref: D0 = {D0:.6f}")
    print(f"  Jump amplification: tau = {tau:.6f}")

    print("\n" + "=" * 80)
    print("V8 MODEL SUMMARY:")
    print("=" * 80)
    print(f"  {len(params)} physics parameters from quartic double-well potential")
    print("  V8 potential: V(mu,T) = h0/L^4 * (mu-mu_L)^2 * (mu-mu_H)^2 - g(T)*(mu-mu_mid)")
    print("  V8 tilt: g(T) = g0 + g_T * dT (temperature dependence via tilt)")
    print("  V8 noise: D(T) = D0 * exp(D_T * dT)")
    print("  V8 escape: Kramers-like pi = 1 - omega0 * exp(-dV/D)")
    print("  V8 drift: d1 = -V'(mu)*dt, d2 = -V'(mu)*tau")
    print("=" * 80)


if __name__ == "__main__":
    main()

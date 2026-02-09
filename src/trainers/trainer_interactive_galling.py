"""
Trainer for Interactive Galling Model

Likelihood-based training that accounts for:
1. Galling density (ρ) as state variable
2. Probabilistic regime transitions
3. Continuous formation/healing competition

Key difference from previous trainer:
- Multi-modal likelihood estimation (low COF vs high COF modes)
- Regime transition timing validation
- Spike interval distribution matching

Author: Claude
Date: 2026-01-15
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class InteractiveGallingTrainer:
    """
    Trains InteractiveGallingModel by maximizing likelihood.

    Uses Monte Carlo sampling with multi-modal density estimation
    to handle the bimodal nature of COF distributions (low vs high).
    """

    def __init__(
        self,
        model,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.

        Args:
            model: InteractiveGallingModel instance
            config: Training configuration
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize log-likelihood
            factor=0.5,
            patience=10
        )

        # Monte Carlo settings
        self.n_samples = config.get('n_likelihood_samples', 30)
        self.bandwidth = config.get('kde_bandwidth', 0.08)

        # Training history
        self.history = {
            'train_log_likelihood': [],
            'physics_params': [],
            'regime_metrics': []
        }

        # Best model tracking
        self.best_log_likelihood = -float('inf')
        self.best_params = None

    def compute_log_likelihood(
        self,
        temp_data_dict: Dict,
        n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Estimate log P(observed_data | params) via Monte Carlo.

        Uses adaptive KDE that handles bimodal distributions better.

        Args:
            temp_data_dict: Dict[temp] -> {'mean_cof': array, 'n_cycles': int}
            n_samples: Number of Monte Carlo samples

        Returns:
            total_log_likelihood: Scalar tensor (with gradients)
            metrics: Dict with regime transition statistics
        """
        if n_samples is None:
            n_samples = self.n_samples

        total_log_likelihood = torch.tensor(0.0, device=self.device)
        metrics = {}

        for temp, temp_data in temp_data_dict.items():
            # Observed COF data
            observed_cof = torch.tensor(
                temp_data['mean_cof'],
                dtype=torch.float32,
                device=self.device
            )
            n_cycles = temp_data['n_cycles']

            # Run stochastic simulations
            simulated_trajs = []
            simulated_betas = []
            simulated_rhos = []

            for _ in range(n_samples):
                sim = self.model.simulate_multiple_cycles(
                    T=temp,
                    n_cycles=n_cycles,
                    M_init=0.0,
                    rho_init=0.0,
                    add_noise=True
                )
                simulated_trajs.append(sim['mu_history'])
                simulated_betas.append(sim['beta_history'])
                simulated_rhos.append(sim['rho_history'])

            # Stack: [n_samples, n_cycles]
            sim_stack = torch.stack(simulated_trajs)
            beta_stack = torch.stack(simulated_betas)
            rho_stack = torch.stack(simulated_rhos)

            # Compute likelihood for each cycle
            temp_log_likelihood = torch.tensor(0.0, device=self.device)

            for n in range(n_cycles):
                # Distribution of simulated COF at cycle n
                sim_values = sim_stack[:, n]

                # Adaptive bandwidth based on local variance
                local_std = sim_values.std().item()
                adaptive_bw = max(self.bandwidth, local_std * 0.5)

                # Gaussian KDE
                distances = (sim_values - observed_cof[n]) ** 2
                densities = torch.exp(-distances / (2 * adaptive_bw**2))
                p_n = densities.mean() + 1e-10

                temp_log_likelihood = temp_log_likelihood + torch.log(p_n)

            total_log_likelihood = total_log_likelihood + temp_log_likelihood

            # Compute regime metrics for this temperature
            mean_beta = beta_stack.mean(dim=0).detach().cpu().numpy()
            mean_rho = rho_stack.mean(dim=0).detach().cpu().numpy()

            # Detect regime transitions (β > 0.5)
            high_regime_mask = mean_beta > 0.5
            n_high_cycles = high_regime_mask.sum()
            transition_cycle = np.argmax(high_regime_mask) if n_high_cycles > 0 else -1

            metrics[temp] = {
                'mean_beta': mean_beta.mean(),
                'mean_rho': mean_rho.mean(),
                'final_rho': mean_rho[-1],
                'n_high_cycles': int(n_high_cycles),
                'first_transition': int(transition_cycle),
                'log_likelihood': temp_log_likelihood.item()
            }

        return total_log_likelihood, metrics

    def compute_trend_loss(
        self,
        temp_data_dict: Dict,
        n_samples: int = 10
    ) -> torch.Tensor:
        """
        Additional loss to match overall trends (mean, variance, etc.).

        This helps when exact cycle-by-cycle matching is difficult.
        """
        trend_loss = torch.tensor(0.0, device=self.device)

        for temp, temp_data in temp_data_dict.items():
            observed_cof = torch.tensor(
                temp_data['mean_cof'],
                dtype=torch.float32,
                device=self.device
            )
            n_cycles = temp_data['n_cycles']

            # Run simulations
            sim_means = []
            sim_stds = []
            sim_maxs = []

            for _ in range(n_samples):
                sim = self.model.simulate_multiple_cycles(
                    T=temp,
                    n_cycles=n_cycles,
                    M_init=0.0,
                    rho_init=0.0,
                    add_noise=True
                )
                mu = sim['mu_history']
                sim_means.append(mu.mean())
                sim_stds.append(mu.std())
                sim_maxs.append(mu.max())

            # Average over simulations
            mean_sim_mean = torch.stack(sim_means).mean()
            mean_sim_std = torch.stack(sim_stds).mean()
            mean_sim_max = torch.stack(sim_maxs).mean()

            # Observed statistics
            obs_mean = observed_cof.mean()
            obs_std = observed_cof.std()
            obs_max = observed_cof.max()

            # Trend matching loss
            trend_loss = trend_loss + (mean_sim_mean - obs_mean) ** 2
            trend_loss = trend_loss + (mean_sim_std - obs_std) ** 2
            trend_loss = trend_loss + 0.5 * (mean_sim_max - obs_max) ** 2

        return trend_loss

    def train(
        self,
        temp_data_dict: Dict,
        n_epochs: int = 100,
        patience_limit: int = 25,
        save_dir: str = 'results/interactive_galling',
        verbose: bool = True
    ):
        """
        Train model via likelihood maximization.

        Args:
            temp_data_dict: Training data
            n_epochs: Maximum epochs
            patience_limit: Early stopping patience
            save_dir: Directory for results
            verbose: Print progress
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if verbose:
            print("=" * 80)
            print("INTERACTIVE GALLING MODEL TRAINING")
            print("=" * 80)
            print(f"Monte Carlo samples: {self.n_samples}")
            print(f"KDE bandwidth: {self.bandwidth}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
            print("=" * 80)

        patience_counter = 0

        for epoch in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Compute log-likelihood
            log_likelihood, metrics = self.compute_log_likelihood(temp_data_dict)

            # Compute trend loss for stability
            trend_loss = self.compute_trend_loss(temp_data_dict, n_samples=5)

            # Combined loss (negative likelihood + trend matching)
            loss = -log_likelihood + 0.5 * trend_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Ensure parameters stay valid
            with torch.no_grad():
                for name, param in self.model.physics_params.items():
                    if name == 'alpha':
                        param.data.clamp_(min=0.01, max=0.99)
                    elif name == 'rho_threshold':
                        param.data.clamp_(min=0.1, max=0.9)
                    else:
                        param.data.clamp_(min=1e-6)

            # Update scheduler
            self.scheduler.step(log_likelihood.item())

            # Record history
            self.history['train_log_likelihood'].append(log_likelihood.item())
            self.history['physics_params'].append(self.model.get_physics_params())
            self.history['regime_metrics'].append(metrics)

            # Track best model
            is_best = False
            if log_likelihood.item() > self.best_log_likelihood:
                self.best_log_likelihood = log_likelihood.item()
                self.best_params = self.model.get_physics_params()
                patience_counter = 0
                is_best = True
                self._save_checkpoint(save_path / 'best_model.pth', epoch, best=True)
            else:
                patience_counter += 1

            # Print progress every epoch (concise one-liner)
            if verbose:
                best_marker = " *BEST*" if is_best else ""
                print(f"[{epoch+1:3d}/{n_epochs}] LL={log_likelihood.item():7.2f} | "
                      f"Trend={trend_loss.item():.4f} | "
                      f"LR={self.optimizer.param_groups[0]['lr']:.5f} | "
                      f"Patience={patience_counter:2d}{best_marker}")

            # Print detailed metrics every 10 epochs
            if verbose and (epoch + 1) % 10 == 0:
                print("-" * 60)
                for temp, m in sorted(metrics.items()):
                    print(f"  {temp}°C: mean_β={m['mean_beta']:.3f}, "
                          f"final_ρ={m['final_rho']:.3f}, "
                          f"high_cycles={m['n_high_cycles']}")
                print("-" * 60)

            # Print parameters every 25 epochs
            if verbose and (epoch + 1) % 25 == 0:
                print("\n  Current parameters:")
                params = self.model.get_physics_params()
                for name, value in params.items():
                    print(f"    {name:15s} = {value:.4f}")
                print()

            # Early stopping
            if patience_counter >= patience_limit:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Save final results
        self._save_checkpoint(save_path / 'final_model.pth', n_epochs, best=False)
        self._save_history(save_path / 'training_history.json')

        if verbose:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)
            print(f"Best log-likelihood: {self.best_log_likelihood:.2f}")
            print("\nBest parameters:")
            for name, value in self.best_params.items():
                print(f"  {name:15s} = {value:.4f}")
            print("=" * 80)

    def _save_checkpoint(self, path: Path, epoch: int, best: bool = False):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'log_likelihood': self.best_log_likelihood if best else self.history['train_log_likelihood'][-1],
            'physics_params': self.model.get_physics_params(),
            'config': self.config
        }, path)

    def _save_history(self, path: Path):
        """Save training history as JSON."""
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            'train_log_likelihood': self.history['train_log_likelihood'],
            'physics_params': self.history['physics_params'],
            'regime_metrics': [
                {temp: {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in m.items()}
                 for temp, m in epoch_metrics.items()}
                for epoch_metrics in self.history['regime_metrics']
            ]
        }
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


def validate_model(
    model,
    temp_data_dict: Dict,
    n_simulations: int = 20,
    save_dir: str = 'results/validation',
    device: str = 'cpu'
):
    """
    Validate trained model by comparing simulations to observed data.

    Args:
        model: Trained InteractiveGallingModel
        temp_data_dict: Observed data
        n_simulations: Number of simulations per temperature
        save_dir: Directory for validation plots
        device: Device for computations
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()

    print("\n" + "=" * 80)
    print("MODEL VALIDATION")
    print("=" * 80)

    for temp, temp_data in sorted(temp_data_dict.items()):
        observed_cof = temp_data['mean_cof']
        n_cycles = temp_data['n_cycles']

        print(f"\n{temp}°C ({n_cycles} cycles):")
        print(f"  Observed: mean={observed_cof.mean():.4f}, "
              f"std={observed_cof.std():.4f}, "
              f"range=[{observed_cof.min():.4f}, {observed_cof.max():.4f}]")

        # Run multiple simulations
        all_sims = []
        all_betas = []
        all_rhos = []

        with torch.no_grad():
            for _ in range(n_simulations):
                sim = model.simulate_multiple_cycles(
                    T=temp,
                    n_cycles=n_cycles,
                    M_init=0.0,
                    rho_init=0.0,
                    add_noise=True
                )
                all_sims.append(sim['mu_history'].cpu().numpy())
                all_betas.append(sim['beta_history'].cpu().numpy())
                all_rhos.append(sim['rho_history'].cpu().numpy())

        all_sims = np.array(all_sims)
        all_betas = np.array(all_betas)
        all_rhos = np.array(all_rhos)

        # Statistics
        sim_mean = all_sims.mean(axis=0)
        sim_std = all_sims.std(axis=0)
        sim_min = all_sims.min(axis=0)
        sim_max = all_sims.max(axis=0)

        print(f"  Simulated: mean={sim_mean.mean():.4f}, "
              f"std={sim_mean.std():.4f}, "
              f"range=[{sim_min.min():.4f}, {sim_max.max():.4f}]")

        # Regime statistics
        mean_beta = all_betas.mean(axis=0)
        mean_rho = all_rhos.mean(axis=0)
        print(f"  Mean β (regime): {mean_beta.mean():.4f}, final: {mean_beta[-1]:.4f}")
        print(f"  Mean ρ (galling density): {mean_rho.mean():.4f}, final: {mean_rho[-1]:.4f}")

        # Create validation plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: COF comparison
        ax = axes[0]
        cycles = np.arange(1, n_cycles + 1)

        # Simulated envelope
        ax.fill_between(cycles, sim_min, sim_max, alpha=0.3, color='orange',
                        label='Simulated range')
        ax.plot(cycles, sim_mean, '-', color='orange', linewidth=2,
                label='Simulated mean')

        # Observed
        ax.plot(cycles, observed_cof, 'o-', color='blue', markersize=2,
                alpha=0.7, label='Observed')

        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Cycle-averaged COF')
        ax.set_title(f'{temp}°C: COF Comparison ({n_simulations} simulations)')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: Regime parameter β
        ax = axes[1]
        beta_min = all_betas.min(axis=0)
        beta_max = all_betas.max(axis=0)

        ax.fill_between(cycles, beta_min, beta_max, alpha=0.3, color='green')
        ax.plot(cycles, mean_beta, '-', color='green', linewidth=2)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
                   label='Galling threshold')

        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Regime Parameter β')
        ax.set_title(f'{temp}°C: Regime Evolution')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 3: Galling density ρ
        ax = axes[2]
        rho_min = all_rhos.min(axis=0)
        rho_max = all_rhos.max(axis=0)

        ax.fill_between(cycles, rho_min, rho_max, alpha=0.3, color='purple')
        ax.plot(cycles, mean_rho, '-', color='purple', linewidth=2)

        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Galling Density ρ')
        ax.set_title(f'{temp}°C: Galling Density Evolution')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / f'validation_{temp}C.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {save_path / f'validation_{temp}C.png'}")

        # Generate single trajectory plot for comparison
        with torch.no_grad():
            single_sim = model.simulate_multiple_cycles(
                T=temp,
                n_cycles=n_cycles,
                M_init=0.0,
                rho_init=0.0,
                add_noise=True
            )

        single_mu = single_sim['mu_history'].cpu().numpy()
        single_beta = single_sim['beta_history'].cpu().numpy()
        single_rho = single_sim['rho_history'].cpu().numpy()

        # Create single trajectory plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: COF - single trajectory vs observed
        ax = axes[0]
        ax.plot(cycles, observed_cof, 'o-', color='blue', markersize=3,
                alpha=0.7, label='Observed', linewidth=1.5)
        ax.plot(cycles, single_mu, 's-', color='orange', markersize=2,
                alpha=0.8, label='Simulated (single run)', linewidth=1.5)

        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Cycle-averaged COF')
        ax.set_title(f'{temp}°C: Single Trajectory Comparison')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: Regime parameter β
        ax = axes[1]
        ax.plot(cycles, single_beta, '-', color='green', linewidth=2)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
                   label='Galling threshold')

        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Regime Parameter β')
        ax.set_title(f'{temp}°C: Regime Evolution (Single Run)')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 3: Galling density ρ
        ax = axes[2]
        ax.plot(cycles, single_rho, '-', color='purple', linewidth=2)

        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Galling Density ρ')
        ax.set_title(f'{temp}°C: Galling Density Evolution (Single Run)')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / f'validation_single_{temp}C.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {save_path / f'validation_single_{temp}C.png'}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


# Testing
if __name__ == "__main__":
    print("Interactive Galling Trainer module loaded successfully!")
    print("\nKey features:")
    print("  - Likelihood-based training with adaptive KDE")
    print("  - Trend matching loss for stability")
    print("  - Regime transition metrics tracking")
    print("  - Comprehensive validation with plots")

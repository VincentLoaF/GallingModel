"""
Training for Pure Physics-Only Model

Simplified training since there's no neural network component.
Only optimizes physics parameters based on cycle-averaged COF.

Author: Claude Code
Date: 2026-01-13
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List

try:
    from ..models.physics_model import GallingPhysicsModel
except ImportError:
    from models.physics_model import GallingPhysicsModel


class PhysicsOnlyTrainer:
    """Trainer for pure physics model (no neural network)"""

    def __init__(self, model: GallingPhysicsModel, config: Dict):
        """
        Initialize trainer.

        Args:
            model: Physics-only model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.model.to(self.device)

        # Setup directories
        self._setup_directories()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'physics_params': []
        }

    def _setup_directories(self):
        """Create output directories"""
        dirs = [
            self.config['output']['model_save_dir'],
            self.config['output']['plot_save_dir'],
            self.config['output']['log_dir']
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def train(self, dataset: List[Dict], verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train physics parameters using cycle-averaged COF.

        Since there's no neural network, we skip Stage 1 and directly
        optimize physics parameters to match observed cycle-averaged COF.

        Args:
            dataset: Dataset from data_loader.create_pytorch_dataset()
            verbose: Print progress

        Returns:
            Dictionary with training history
        """
        if verbose:
            print("\n" + "=" * 70)
            print("TRAINING PURE PHYSICS MODEL")
            print("Optimizing physics parameters only (no neural network)")
            print("=" * 70)

        # Configuration
        cfg = self.config['training']['stage2']  # Use stage2 config
        n_epochs = cfg['n_epochs']

        # Split dataset
        torch.manual_seed(self.config['validation']['random_seed'])
        train_size = int(self.config['validation']['train_val_split'] * len(dataset))
        val_size = len(dataset) - train_size

        indices = torch.randperm(len(dataset)).tolist()
        train_dataset = [dataset[i] for i in indices[:train_size]]
        val_dataset = [dataset[i] for i in indices[train_size:]]

        if verbose:
            print(f"\nDataset split: {train_size} train, {val_size} validation")
            print(f"Physics parameters to optimize: {len(self.model.physics_params)}")

        # Optimizer (only physics parameters)
        optimizer = Adam(
            self.model.physics_params.parameters(),
            lr=cfg['learning_rate_physics'],
            weight_decay=self.config['training']['stage1']['weight_decay']
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg['scheduler_nn']['factor'],
            patience=cfg['scheduler_nn']['patience'],
            min_lr=cfg['scheduler_nn']['min_lr']
        )

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = cfg['early_stopping']['patience']
        min_delta = cfg['early_stopping']['min_delta']

        # Training loop
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = self._train_epoch(train_dataset, optimizer)

            # Validation
            self.model.eval()
            val_loss = self._validate_epoch(val_dataset)

            # Update scheduler
            scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if (epoch + 1) % self.config['output']['print_physics_params_every'] == 0:
                self.history['physics_params'].append({
                    'epoch': epoch + 1,
                    **self.model.get_physics_params()
                })

            # Print progress
            if verbose and (epoch + 1) % self.config['output']['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {lr:.2e}")

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                if self.config['output']['save_best_only']:
                    save_path = Path(self.config['output']['model_save_dir']) / 'best_model.pth'
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'physics_params': self.model.get_physics_params(),
                        'epoch': epoch + 1,
                        'val_loss': val_loss
                    }, save_path)
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

            # Save checkpoints
            if (epoch + 1) % self.config['output']['save_every_n_epochs'] == 0:
                save_path = Path(self.config['output']['model_save_dir']) / f'checkpoint_epoch{epoch+1}.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'physics_params': self.model.get_physics_params(),
                    'epoch': epoch + 1,
                    'val_loss': val_loss
                }, save_path)

        if verbose:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print("\nFitted Physics Parameters:")
            for name, value in self.model.get_physics_params().items():
                print(f"  {name:10s} = {value:.6f}")
            print("=" * 70)

        return self.history

    def _train_epoch(self, dataset: List[Dict], optimizer) -> float:
        """Train for one epoch"""
        total_loss = 0.0
        num_samples = 0

        # Group by temperature for sequential simulation
        temp_groups = {}
        for item in dataset:
            temp = item['T']
            if temp not in temp_groups:
                temp_groups[temp] = []
            temp_groups[temp].append(item)

        for temp, cycles_data in temp_groups.items():
            # Get features for all cycles at this temperature
            features_list = [item['features'].to(self.device) for item in cycles_data]

            # Run multi-cycle simulation
            output = self.model.forward_multi_cycle(temp, features_list, M_init=0.0)

            # Compute loss on cycle-averaged COF
            mu_mean_pred = output['mu_mean_history']
            mu_mean_obs = torch.tensor(
                [item['mean_cof_observed'] for item in cycles_data],
                device=self.device
            )

            # MSE loss
            loss = torch.mean((mu_mean_pred - mu_mean_obs) ** 2)

            # Regularization (keep parameters reasonable)
            reg_loss = 0.0
            for param in self.model.physics_params.values():
                reg_loss += torch.abs(param)

            w_reg = self.config['training']['stage2']['loss_weights']['w_regularization']
            total_loss_batch = loss + w_reg * reg_loss

            # Backward pass
            self.model.zero_grad()
            total_loss_batch.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['stage2']['grad_clip_norm']
            )

            # Optimizer step
            optimizer.step()

            total_loss += total_loss_batch.item() * len(cycles_data)
            num_samples += len(cycles_data)

        return total_loss / num_samples if num_samples > 0 else 0.0

    def _validate_epoch(self, dataset: List[Dict]) -> float:
        """Validate for one epoch"""
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            # Group by temperature
            temp_groups = {}
            for item in dataset:
                temp = item['T']
                if temp not in temp_groups:
                    temp_groups[temp] = []
                temp_groups[temp].append(item)

            for temp, cycles_data in temp_groups.items():
                features_list = [item['features'].to(self.device) for item in cycles_data]

                # Run multi-cycle simulation
                output = self.model.forward_multi_cycle(temp, features_list, M_init=0.0)

                # Compute loss
                mu_mean_pred = output['mu_mean_history']
                mu_mean_obs = torch.tensor(
                    [item['mean_cof_observed'] for item in cycles_data],
                    device=self.device
                )

                loss = torch.mean((mu_mean_pred - mu_mean_obs) ** 2)

                total_loss += loss.item() * len(cycles_data)
                num_samples += len(cycles_data)

        return total_loss / num_samples if num_samples > 0 else 0.0

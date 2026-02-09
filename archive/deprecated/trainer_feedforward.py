"""
Training loops for PINN model.

Implements two-stage training:
1. Stage 1: Pre-train neural network on in-cycle data (supervised)
2. Stage 2: Joint training of NN + physics parameters (physics-informed)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time

try:
    from ..models.pinn_feedforward import GallingPINN
    from ..utils.temporal_split import temporal_split_per_temperature, validate_no_leakage
except ImportError:
    from models.pinn_feedforward import GallingPINN
    from utils.temporal_split import temporal_split_per_temperature, validate_no_leakage


def collate_variable_length(batch):
    """
    Custom collate function for variable-length sequences.

    Since different cycles may have different numbers of timesteps after preprocessing,
    we cannot use the default stacking behavior. Instead, we keep them as lists.
    """
    # Don't stack - keep as lists since sequences have different lengths
    return {
        'M': [item['M'] for item in batch],
        'T': torch.tensor([item['T'] for item in batch]),
        'features': [item['features'] for item in batch],
        'target_cof': [item['target_cof'] for item in batch],
        'cycle_num': [item['cycle_num'] for item in batch],
        'temp': [item['temp'] for item in batch],
        'mean_cof_observed': [item['mean_cof_observed'] for item in batch]
    }


class GallingDataset(Dataset):
    """PyTorch Dataset wrapper for galling data"""

    def __init__(self, data_list: List[Dict]):
        """
        Initialize dataset.

        Args:
            data_list: List of dicts from data_loader.create_pytorch_dataset()
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PINNTrainer:
    """Training orchestrator for PINN model"""

    def __init__(
        self,
        model: GallingPINN,
        config: Dict,
        device: str = None
    ):
        """
        Initialize trainer.

        Args:
            model: PINN model instance
            config: Configuration dictionary
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Create output directories
        self._setup_directories()

        # Loss function
        self.criterion_mse = nn.MSELoss()

        # Training history
        self.history = {
            'stage1': {'train_loss': [], 'val_loss': []},
            'stage2': {'train_loss': [], 'val_loss': [],
                      'loss_incycle': [], 'loss_physics': [], 'loss_reg': []}
        }

    def _setup_directories(self):
        """Create output directories if they don't exist"""
        dirs = [
            self.config['output']['model_save_dir'],
            self.config['output']['plot_save_dir'],
            self.config['output']['log_dir']
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def stage1_pretrain_nn(
        self,
        dataset: List[Dict],
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Stage 1: Pre-train neural network on in-cycle data.

        Freeze physics parameters, optimize only NN weights using supervised learning.
        Target: Minimize MSE between predicted and observed COF(t).

        Args:
            dataset: Dataset from data_loader.create_pytorch_dataset()
            verbose: Print progress

        Returns:
            Dictionary with training history
        """
        if verbose:
            print("\n" + "=" * 70)
            print("STAGE 1: Pre-training Neural Network (Supervised Learning)")
            print("=" * 70)

        # Freeze physics parameters
        for param in self.model.physics_params.parameters():
            param.requires_grad = False

        # Configuration
        cfg = self.config['training']['stage1']
        n_epochs = cfg['n_epochs']
        batch_size = cfg['batch_size']

        # Split dataset temporally (no data leakage!)
        train_ratio = self.config['validation']['train_val_split']
        train_data, val_data = temporal_split_per_temperature(
            dataset,
            train_ratio=train_ratio,
            verbose=verbose
        )

        # Validate no temporal leakage
        validate_no_leakage(train_data, val_data)

        # Wrap in PyTorch Dataset
        train_dataset = GallingDataset(train_data)
        val_dataset = GallingDataset(val_data)

        # Data loaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_variable_length
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_variable_length
        )

        # Optimizer and scheduler
        optimizer = Adam(
            self.model.in_cycle_nn.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg['scheduler']['factor'],
            patience=cfg['scheduler']['patience'],
            min_lr=cfg['scheduler']['min_lr']
        )

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = cfg['early_stopping']['patience']
        min_delta = cfg['early_stopping']['min_delta']

        # Training loop
        for epoch in range(n_epochs):
            epoch_start = time.time()

            # Training
            self.model.train()
            train_loss = 0.0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]",
                            disable=not verbose)

            for batch in train_pbar:
                optimizer.zero_grad()

                # Process each item in batch separately (variable length sequences)
                batch_loss = 0.0
                for i in range(len(batch['features'])):
                    features = batch['features'][i].to(self.device)  # [time, 8]
                    target_cof = batch['target_cof'][i].to(self.device)  # [time]
                    M = torch.tensor(batch['M'][i], device=self.device)  # Placeholder

                    # Forward pass
                    cof_pred = self.model.forward_incycle(M, features.unsqueeze(0))  # Add batch dim
                    cof_pred = cof_pred.squeeze(0)  # Remove batch dim

                    # Loss for this item
                    loss = self.criterion_mse(cof_pred, target_cof)
                    batch_loss += loss

                # Average loss over batch
                batch_loss = batch_loss / len(batch['features'])

                # Backward
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.item()
                train_pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})

            train_loss /= len(train_loader)

            # Validation
            val_loss = self._validate_stage1(val_loader)

            # Update history
            self.history['stage1']['train_loss'].append(train_loss)
            self.history['stage1']['val_loss'].append(val_loss)

            # Scheduler step
            scheduler.step(val_loss)

            # Logging
            epoch_time = time.time() - epoch_start
            if verbose and (epoch + 1) % self.config['output']['log_interval'] == 0:
                print(f"\nEpoch {epoch+1}/{n_epochs}:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {val_loss:.6f}")
                print(f"  Time: {epoch_time:.2f}s")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Save best model
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0

                if self.config['output']['save_best_only']:
                    save_path = Path(self.config['output']['model_save_dir']) / 'best_stage1.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, save_path)
                    if verbose:
                        print(f"  → Saved best model (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"\nEarly stopping triggered (patience={early_stop_patience})")
                break

        if verbose:
            print(f"\nStage 1 complete. Best val loss: {best_val_loss:.6f}")

        return self.history['stage1']

    def _validate_stage1(self, val_loader: DataLoader) -> float:
        """Validation for Stage 1"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Process each item in batch separately (variable length sequences)
                batch_loss = 0.0
                for i in range(len(batch['features'])):
                    features = batch['features'][i].to(self.device)
                    target_cof = batch['target_cof'][i].to(self.device)
                    M = torch.tensor(batch['M'][i], device=self.device)

                    cof_pred = self.model.forward_incycle(M, features.unsqueeze(0))
                    cof_pred = cof_pred.squeeze(0)

                    loss = self.criterion_mse(cof_pred, target_cof)
                    batch_loss += loss

                # Average loss over batch
                batch_loss = batch_loss / len(batch['features'])
                val_loss += batch_loss.item()

        return val_loss / len(val_loader)

    def _validate_stage2(
        self,
        val_temp_groups: Dict[float, List[Dict]],
        w1: float,
        w2: float,
        w3: float
    ) -> float:
        """
        Validation for Stage 2 (physics-informed training).

        Args:
            val_temp_groups: Validation data grouped by temperature
            w1: Weight for in-cycle loss
            w2: Weight for physics loss
            w3: Weight for regularization loss

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for temp, cycles_data in val_temp_groups.items():
                # Prepare data
                features_list = [c['features'].to(self.device) for c in cycles_data]
                target_cof_list = [c['target_cof'].to(self.device) for c in cycles_data]
                mean_cof_observed = torch.tensor(
                    [c['mean_cof_observed'] for c in cycles_data],
                    dtype=torch.float32,
                    device=self.device
                )

                # Multi-cycle forward pass
                output = self.model.forward_multi_cycle(temp, features_list, M_init=0.0)

                cof_predicted = output['cof_predicted']
                mu_mean_predicted = output['mu_mean_history']

                # Loss 1: In-cycle COF prediction
                loss_1 = 0.0
                for pred, target in zip(cof_predicted, target_cof_list):
                    loss_1 += self.criterion_mse(pred, target)
                loss_1 /= len(cof_predicted)

                # Loss 2: Cycle-averaged COF consistency
                loss_2 = self.criterion_mse(mu_mean_predicted, mean_cof_observed)

                # Loss 3: Physics regularization
                loss_3 = self._physics_regularization()

                # Combined loss
                loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

                total_val_loss += loss.item()

        return total_val_loss / len(val_temp_groups)

    def stage2_joint_training(
        self,
        dataset: List[Dict],
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Stage 2: Joint physics-informed training.

        Optimize both NN and physics parameters together.
        Loss combines:
        1. In-cycle COF prediction error
        2. Cycle-averaged COF consistency (physics constraint)
        3. Physics parameter regularization

        Args:
            dataset: Dataset from data_loader.create_pytorch_dataset()
            verbose: Print progress

        Returns:
            Dictionary with training history
        """
        if verbose:
            print("\n" + "=" * 70)
            print("STAGE 2: Joint Physics-Informed Training")
            print("=" * 70)

        # Unfreeze physics parameters
        for param in self.model.physics_params.parameters():
            param.requires_grad = True

        # Configuration
        cfg = self.config['training']['stage2']
        n_epochs = cfg['n_epochs']

        # Split dataset temporally (same as Stage 1!)
        train_ratio = self.config['validation']['train_val_split']
        train_data, val_data = temporal_split_per_temperature(
            dataset,
            train_ratio=train_ratio,
            verbose=verbose
        )

        # Validate no temporal leakage
        validate_no_leakage(train_data, val_data)

        # Group TRAINING data by temperature for multi-cycle simulation
        train_temp_groups = self._group_by_temperature(train_data)
        val_temp_groups = self._group_by_temperature(val_data)

        if verbose:
            print(f"\nTraining temperature groups:")
            for temp, cycles in train_temp_groups.items():
                print(f"  {temp}°C: {len(cycles)} cycles")
            print(f"\nValidation temperature groups:")
            for temp, cycles in val_temp_groups.items():
                print(f"  {temp}°C: {len(cycles)} cycles")

        # Separate optimizers for NN and physics
        optimizer_nn = Adam(
            self.model.in_cycle_nn.parameters(),
            lr=cfg['learning_rate_nn']
        )

        optimizer_physics = Adam(
            self.model.physics_params.parameters(),
            lr=cfg['learning_rate_physics']
        )

        # Scheduler for NN
        scheduler_nn = ReduceLROnPlateau(
            optimizer_nn,
            mode='min',
            factor=cfg['scheduler_nn']['factor'],
            patience=cfg['scheduler_nn']['patience'],
            min_lr=cfg['scheduler_nn']['min_lr']
        )

        # Loss weights
        w1 = cfg['loss_weights']['w_incycle']
        w2 = cfg['loss_weights']['w_physics']
        w3 = cfg['loss_weights']['w_regularization']

        # Early stopping
        best_total_loss = float('inf')
        patience_counter = 0
        early_stop_patience = cfg['early_stopping']['patience']
        min_delta = cfg['early_stopping']['min_delta']

        # Training loop
        for epoch in range(n_epochs):
            epoch_start = time.time()
            self.model.train()

            total_loss = 0.0
            total_loss_incycle = 0.0
            total_loss_physics = 0.0
            total_loss_reg = 0.0

            # Train on each temperature group (TRAINING SET ONLY)
            for temp, cycles_data in train_temp_groups.items():
                # Prepare data
                features_list = [c['features'].to(self.device) for c in cycles_data]
                target_cof_list = [c['target_cof'].to(self.device) for c in cycles_data]
                mean_cof_observed = torch.tensor(
                    [c['mean_cof_observed'] for c in cycles_data],
                    dtype=torch.float32,
                    device=self.device
                )

                # Multi-cycle forward pass
                output = self.model.forward_multi_cycle(temp, features_list, M_init=0.0)

                cof_predicted = output['cof_predicted']
                mu_mean_predicted = output['mu_mean_history']

                # Loss 1: In-cycle COF prediction
                loss_1 = 0.0
                for pred, target in zip(cof_predicted, target_cof_list):
                    loss_1 += self.criterion_mse(pred, target)
                loss_1 /= len(cof_predicted)

                # Loss 2: Cycle-averaged COF consistency
                loss_2 = self.criterion_mse(mu_mean_predicted, mean_cof_observed)

                # Loss 3: Physics regularization
                loss_3 = self._physics_regularization()

                # Combined loss
                loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

                # Backward
                optimizer_nn.zero_grad()
                optimizer_physics.zero_grad()

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=cfg['grad_clip_norm']
                )

                optimizer_nn.step()
                optimizer_physics.step()

                # Accumulate losses
                total_loss += loss.item()
                total_loss_incycle += loss_1.item()
                total_loss_physics += loss_2.item()
                total_loss_reg += loss_3.item()

            # Average over temperature groups
            n_temps = len(train_temp_groups)
            total_loss /= n_temps
            total_loss_incycle /= n_temps
            total_loss_physics /= n_temps
            total_loss_reg /= n_temps

            # Validation on VALIDATION SET (not training set!)
            val_loss = self._validate_stage2(val_temp_groups, w1, w2, w3)

            # Update history
            self.history['stage2']['train_loss'].append(total_loss)
            self.history['stage2']['val_loss'].append(val_loss)
            self.history['stage2']['loss_incycle'].append(total_loss_incycle)
            self.history['stage2']['loss_physics'].append(total_loss_physics)
            self.history['stage2']['loss_reg'].append(total_loss_reg)

            # Scheduler step (use validation loss!)
            scheduler_nn.step(val_loss)

            # Logging
            epoch_time = time.time() - epoch_start
            if verbose and (epoch + 1) % self.config['output']['log_interval'] == 0:
                print(f"\nEpoch {epoch+1}/{n_epochs}:")
                print(f"  Train Loss:    {total_loss:.6f}")
                print(f"  Val Loss:      {val_loss:.6f}")
                print(f"  In-Cycle Loss: {total_loss_incycle:.6f}")
                print(f"  Physics Loss:  {total_loss_physics:.6f}")
                print(f"  Reg Loss:      {total_loss_reg:.6f}")
                print(f"  Time: {epoch_time:.2f}s")

            # Print physics parameters
            if verbose and (epoch + 1) % self.config['output']['print_physics_params_every'] == 0:
                self._print_physics_params()

            # Save checkpoint
            if (epoch + 1) % self.config['output']['save_every_n_epochs'] == 0:
                self._save_checkpoint(epoch, val_loss, stage=2)

            # Save best model (based on VALIDATION loss!)
            if val_loss < best_total_loss - min_delta:
                best_total_loss = val_loss
                patience_counter = 0

                if self.config['output']['save_best_only']:
                    self._save_checkpoint(epoch, val_loss, stage=2, best=True)
                    if verbose:
                        print(f"  → Saved best model (val loss: {val_loss:.6f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"\nEarly stopping triggered (patience={early_stop_patience})")
                break

        if verbose:
            print(f"\nStage 2 complete. Best total loss: {best_total_loss:.6f}")
            print("\nFinal physics parameters:")
            self._print_physics_params()

        return self.history['stage2']

    def _group_by_temperature(self, dataset: List[Dict]) -> Dict[float, List[Dict]]:
        """Group dataset by temperature"""
        groups = {}
        for item in dataset:
            temp = item['temp']
            if temp not in groups:
                groups[temp] = []
            groups[temp].append(item)
        return groups

    def _physics_regularization(self) -> torch.Tensor:
        """
        Regularization to keep physics parameters in reasonable ranges.

        Constraints:
        - k0 > 0 (attachment rate must be positive)
        - T0, Tc > 0 (temperature sensitivities must be positive)
        - kw ∈ [0, 1] (wear rate in reasonable range)
        - alpha ∈ [0, 1] (retention fraction between 0 and 1)
        """
        penalty = torch.tensor(0.0, device=self.device)

        # k0 > 0
        penalty += torch.relu(-self.model.physics_params['k0']) * 10.0

        # T0, Tc > 0
        penalty += torch.relu(-self.model.physics_params['T0']) * 10.0
        penalty += torch.relu(-self.model.physics_params['Tc']) * 10.0

        # alpha ∈ [0, 1]
        penalty += torch.relu(self.model.physics_params['alpha'] - 1.0) * 10.0
        penalty += torch.relu(-self.model.physics_params['alpha']) * 10.0

        # kw ∈ [0, 1]
        penalty += torch.relu(self.model.physics_params['kw'] - 1.0) * 5.0
        penalty += torch.relu(-self.model.physics_params['kw']) * 10.0

        # M0 > 0
        penalty += torch.relu(-self.model.physics_params['M0']) * 10.0

        return penalty

    def _print_physics_params(self):
        """Print current physics parameter values"""
        params = self.model.get_physics_params()
        print("\n  Physics Parameters:")
        for name, value in params.items():
            print(f"    {name:6s}: {value:8.4f}")

    def _save_checkpoint(self, epoch: int, loss: float, stage: int, best: bool = False):
        """Save model checkpoint"""
        if best:
            filename = f'best_stage{stage}.pth'
        else:
            filename = f'checkpoint_stage{stage}_epoch{epoch+1}.pth'

        save_path = Path(self.config['output']['model_save_dir']) / filename

        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'physics_params': self.model.get_physics_params(),
            'config': self.config
        }

        torch.save(checkpoint, save_path)

    def save_training_history(self, filepath: str = None):
        """Save training history to JSON"""
        if filepath is None:
            filepath = Path(self.config['output']['log_dir']) / 'training_history.json'

        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for stage, metrics in self.history.items():
            history_serializable[stage] = {
                key: [float(x) for x in values]
                for key, values in metrics.items()
            }

        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2)

        print(f"Training history saved to {filepath}")

    def save_final_physics_params(self, filepath: str = None):
        """Save final fitted physics parameters to JSON"""
        if filepath is None:
            filepath = Path(self.config['output']['model_save_dir']) / 'fitted_parameters.json'

        params = self.model.get_physics_params()

        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"Physics parameters saved to {filepath}")

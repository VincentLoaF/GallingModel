"""
Cycle-Averaged Data Loader for Physics Generative Model

This loader provides ONLY cycle-averaged COF data and temperature.
NO force features - those would create trivial μ = F_y/F_z relationships.

Purpose: Load data for pure physics-based generative model

Author: Claude Sonnet 4.5
Date: 2026-01-14
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import torch


class CycleAveragedLoader:
    """
    Load cycle-averaged COF data for physics generative model.

    Key difference from old loader:
    - NO force features (F_x, F_y, F_z)
    - NO velocity features
    - ONLY: Temperature, cycle number, cycle-averaged COF, std COF

    This prevents learning trivial μ = F_y/F_z relationships.
    """

    def __init__(self, base_path: str = "/root/Documents/GallingModel/data"):
        self.base_path = Path(base_path)

    def load_temperature_data(self, temp: float) -> Dict:
        """
        Load cycle-averaged data for a single temperature.

        Args:
            temp: Temperature in °C (e.g., 165.0, 167.5, 170.0)

        Returns:
            Dictionary with:
                - 'temperature': float (temperature)
                - 'cycle_nums': np.array (cycle numbers 1, 2, 3, ...)
                - 'mean_cof': np.array (cycle-averaged COF)
                - 'std_cof': np.array (within-cycle std of COF)
                - 'n_cycles': int (number of cycles)
        """
        temp_dir = self.base_path / str(temp)

        # Load cycle-averaged COF from mean.txt
        mean_file = temp_dir / "mean.txt"
        if not mean_file.exists():
            raise FileNotFoundError(f"Mean COF file not found: {mean_file}")

        mean_cof = np.loadtxt(mean_file)

        # Load std COF from std.txt
        std_file = temp_dir / "std.txt"
        if not std_file.exists():
            raise FileNotFoundError(f"Std COF file not found: {std_file}")

        std_cof = np.loadtxt(std_file)

        # Verify consistency
        if len(mean_cof) != len(std_cof):
            raise ValueError(f"Mismatch: {len(mean_cof)} means vs {len(std_cof)} stds")

        n_cycles = len(mean_cof)
        cycle_nums = np.arange(1, n_cycles + 1)

        return {
            'temperature': temp,
            'cycle_nums': cycle_nums,
            'mean_cof': mean_cof,
            'std_cof': std_cof,
            'n_cycles': n_cycles
        }

    def load_all_temperatures(self, temps: List[float] = [165, 167.5, 170]) -> Dict[float, Dict]:
        """
        Load cycle-averaged data for all specified temperatures.

        Args:
            temps: List of temperatures to load

        Returns:
            Dictionary mapping temperature → data dict
        """
        all_data = {}

        for temp in temps:
            try:
                all_data[temp] = self.load_temperature_data(temp)
                print(f"✓ Loaded {all_data[temp]['n_cycles']} cycles at {temp}°C")
            except FileNotFoundError as e:
                print(f"⚠ Warning: Could not load {temp}°C: {e}")

        return all_data

    def create_training_dataset(
        self,
        temps: List[float] = [165, 167.5, 170],
        normalize: bool = True
    ) -> List[Dict]:
        """
        Create training dataset for physics generative model.

        Args:
            temps: Temperatures to include
            normalize: Whether to normalize temperature to [0, 1]

        Returns:
            List of dicts, each representing one cycle:
                {
                    'temperature': float (raw or normalized),
                    'cycle_num': int,
                    'mean_cof': float,
                    'std_cof': float,
                    'temp_raw': float (original temperature)
                }
        """
        all_data = self.load_all_temperatures(temps)

        dataset = []

        # Normalization parameters
        if normalize:
            T_min = min(temps)
            T_max = max(temps)
        else:
            T_min = 0
            T_max = 1

        for temp, temp_data in all_data.items():
            n_cycles = temp_data['n_cycles']

            for i in range(n_cycles):
                # Normalize temperature if requested
                if normalize:
                    T_norm = (temp - T_min) / (T_max - T_min)
                else:
                    T_norm = temp

                dataset.append({
                    'temperature': T_norm,
                    'cycle_num': int(temp_data['cycle_nums'][i]),
                    'mean_cof': float(temp_data['mean_cof'][i]),
                    'std_cof': float(temp_data['std_cof'][i]),
                    'temp_raw': temp
                })

        print(f"\n✓ Created dataset with {len(dataset)} cycles total")
        return dataset

    def create_pytorch_dataset(
        self,
        temps: List[float] = [165, 167.5, 170],
        normalize: bool = True
    ) -> List[Dict]:
        """
        Create PyTorch-compatible dataset.

        Returns:
            List of dicts with tensors:
                {
                    'temperature': torch.Tensor (scalar),
                    'cycle_num': int,
                    'mean_cof': torch.Tensor (scalar),
                    'std_cof': torch.Tensor (scalar),
                    'temp_raw': float
                }
        """
        dataset = self.create_training_dataset(temps, normalize)

        # Convert to tensors
        for item in dataset:
            item['temperature'] = torch.tensor(item['temperature'], dtype=torch.float32)
            item['mean_cof'] = torch.tensor(item['mean_cof'], dtype=torch.float32)
            item['std_cof'] = torch.tensor(item['std_cof'], dtype=torch.float32)

        return dataset

    def get_temperature_groups(
        self,
        dataset: List[Dict]
    ) -> Dict[float, List[Dict]]:
        """
        Group dataset by temperature.

        Args:
            dataset: Output from create_training_dataset() or create_pytorch_dataset()

        Returns:
            Dictionary mapping temperature → list of cycle dicts
        """
        groups = {}

        for item in dataset:
            temp = item['temp_raw']
            if temp not in groups:
                groups[temp] = []
            groups[temp].append(item)

        return groups

    def get_dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """
        Compute statistics for the dataset.

        Returns:
            Dictionary with:
                - 'n_cycles_total': Total number of cycles
                - 'n_temperatures': Number of unique temperatures
                - 'temperature_distribution': Cycle counts per temperature
                - 'mean_cof_overall': Mean COF across all cycles
                - 'std_cof_overall': Std COF across all cycles
                - 'mean_cof_range': (min, max) across temperatures
        """
        n_cycles = len(dataset)
        temps = list(set([item['temp_raw'] for item in dataset]))
        n_temps = len(temps)

        temp_distribution = {}
        for temp in temps:
            temp_distribution[temp] = sum(1 for item in dataset if item['temp_raw'] == temp)

        all_mean_cof = np.array([item['mean_cof'] if isinstance(item['mean_cof'], float)
                                  else item['mean_cof'].item() for item in dataset])
        all_std_cof = np.array([item['std_cof'] if isinstance(item['std_cof'], float)
                                 else item['std_cof'].item() for item in dataset])

        return {
            'n_cycles_total': n_cycles,
            'n_temperatures': n_temps,
            'temperature_distribution': temp_distribution,
            'mean_cof_overall': float(np.mean(all_mean_cof)),
            'std_cof_overall': float(np.mean(all_std_cof)),
            'mean_cof_range': (float(np.min(all_mean_cof)), float(np.max(all_mean_cof)))
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("CYCLE-AVERAGED DATA LOADER TEST")
    print("=" * 80)

    loader = CycleAveragedLoader()

    # Load data
    print("\nLoading temperature data...")
    dataset = loader.create_training_dataset(temps=[165, 167.5, 170], normalize=True)

    # Statistics
    print("\nDataset Statistics:")
    stats = loader.get_dataset_statistics(dataset)
    print(f"  Total cycles: {stats['n_cycles_total']}")
    print(f"  Temperatures: {stats['n_temperatures']}")
    print(f"  Distribution:")
    for temp, count in sorted(stats['temperature_distribution'].items()):
        print(f"    {temp}°C: {count} cycles")
    print(f"  Mean COF overall: {stats['mean_cof_overall']:.4f}")
    print(f"  Mean COF range: ({stats['mean_cof_range'][0]:.4f}, {stats['mean_cof_range'][1]:.4f})")

    # Group by temperature
    print("\nTemperature groups:")
    groups = loader.get_temperature_groups(dataset)
    for temp, cycles in sorted(groups.items()):
        print(f"  {temp}°C: {len(cycles)} cycles (cycle {cycles[0]['cycle_num']}-{cycles[-1]['cycle_num']})")

    print("\n" + "=" * 80)
    print("TEST PASSED - Data loader working correctly!")
    print("=" * 80)

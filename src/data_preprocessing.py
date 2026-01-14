"""
Data loader for high-frequency (125Hz) galling experimental data.

Handles loading, preprocessing, and PyTorch dataset creation for the PINN model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch


class HighFrequencyDataLoader:
    """Load and preprocess 125Hz experimental data with proper filtering"""

    def __init__(self, base_path: str = "/root/Documents/GallingModel/data"):
        """
        Initialize data loader.

        Args:
            base_path: Root directory containing temperature folders (data/)
        """
        self.base_path = Path(base_path)

        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

    def load_cycle_csv(self, temp: float, cycle_num: int) -> pd.DataFrame:
        """
        Load single cycle CSV file.

        Args:
            temp: Temperature (°C)
            cycle_num: Cycle number (1-indexed)

        Returns:
            DataFrame with columns:
                timestamp, force_x, force_y, force_z, position_x, position_y,
                position_z, timestamp_0, timestep, velocity_x, cof, sliding_distance

        Raises:
            FileNotFoundError: If no matching file is found
        """
        # Handle temperature folder naming (165, 167.5, 170)
        temp_folder = str(temp).replace('.', '_') if '.' in str(temp) else str(int(temp))

        # Try different possible folder names
        possible_folders = [
            self.base_path / str(temp),
            self.base_path / str(int(temp)),
            self.base_path / temp_folder
        ]

        data_folder = None
        for folder in possible_folders:
            if (folder / "data").exists():
                data_folder = folder / "data"
                break

        if data_folder is None:
            raise FileNotFoundError(
                f"No data folder found for {temp}°C. Tried: {[str(f) for f in possible_folders]}"
            )

        # Find CSV file (format: XXXXXX_*.csv where XXXXXX is 6-digit cycle number)
        pattern = f"{cycle_num:06d}_*.csv"
        files = list(data_folder.glob(pattern))

        if len(files) == 0:
            raise FileNotFoundError(
                f"No data found for {temp}°C, cycle {cycle_num} in {data_folder}"
            )

        # Load first matching file
        df = pd.read_csv(files[0])

        return df

    def preprocess_cycle(
        self,
        df: pd.DataFrame,
        min_distance: float = 0.1
    ) -> pd.DataFrame:
        """
        Preprocess cycle data with filtering and feature engineering.

        Steps:
        1. Remove rows with sliding_distance < min_distance (mm) - removes initialization artifacts
        2. Reset index
        3. Add cycle_phase feature [0, 1] marking progress through cycle
        4. Add normalized sliding distance

        Args:
            df: Raw cycle DataFrame
            min_distance: Minimum sliding distance threshold (mm)

        Returns:
            Preprocessed DataFrame with additional columns:
                - cycle_phase: normalized position in cycle [0, 1]
                - sliding_distance_norm: normalized sliding distance

        Raises:
            ValueError: If no valid data remains after filtering
        """
        # Filter out initialization artifacts (sliding_distance < 0.1mm)
        df_clean = df[df['sliding_distance'] >= min_distance].copy()

        if len(df_clean) == 0:
            raise ValueError(
                f"No valid data after filtering (min_distance={min_distance}mm). "
                f"Original length: {len(df)}, all sliding_distance < {min_distance}"
            )

        # Reset index
        df_clean = df_clean.reset_index(drop=True)

        # Normalize cycle phase [0, 1] - marks progress through slide
        df_clean['cycle_phase'] = np.linspace(0, 1, len(df_clean))

        # Normalize sliding distance to [0, 1]
        max_dist = df_clean['sliding_distance'].max()
        if max_dist > 0:
            df_clean['sliding_distance_norm'] = df_clean['sliding_distance'] / max_dist
        else:
            df_clean['sliding_distance_norm'] = 0.0

        return df_clean

    def load_all_cycles_for_temperature(self, temp: float) -> Dict:
        """
        Load all cycles for a given temperature.

        Args:
            temp: Temperature (°C)

        Returns:
            Dictionary with keys:
                - 'data': List of preprocessed DataFrames (one per cycle)
                - 'mean_cof': np.array from mean.txt
                - 'std_cof': np.array from std.txt
                - 'n_cycles': int, number of cycles
                - 'temp': float, temperature

        Raises:
            FileNotFoundError: If temperature folder or summary files not found
        """
        # Find temperature folder
        temp_folder = str(temp).replace('.', '_') if '.' in str(temp) else str(int(temp))
        possible_folders = [
            self.base_path / str(temp),
            self.base_path / str(int(temp)),
            self.base_path / temp_folder
        ]

        temp_path = None
        for folder in possible_folders:
            if folder.exists():
                temp_path = folder
                break

        if temp_path is None:
            raise FileNotFoundError(f"Temperature folder not found for {temp}°C")

        # Find all CSV files in data/ subfolder
        data_folder = temp_path / "data"
        if data_folder.exists():
            data_files = sorted(data_folder.glob("*.csv"))
        else:
            data_files = []

        n_cycles = len(data_files)

        # Load cycle-by-cycle data
        cycle_data = []
        failed_cycles = []

        for i, file_path in enumerate(data_files):
            try:
                df = pd.read_csv(file_path)
                df_clean = self.preprocess_cycle(df, min_distance=0.1)
                cycle_data.append(df_clean)
            except ValueError as e:
                # Skip cycles that fail preprocessing
                failed_cycles.append((i + 1, str(e)))
                continue

        if failed_cycles:
            print(f"Warning: {len(failed_cycles)} cycles failed preprocessing:")
            for cycle_num, error in failed_cycles[:5]:  # Show first 5
                print(f"  Cycle {cycle_num}: {error}")

        # Load summary statistics
        mean_file = temp_path / "mean.txt"
        std_file = temp_path / "std.txt"

        if not mean_file.exists():
            raise FileNotFoundError(f"mean.txt not found in {temp_path}")
        if not std_file.exists():
            raise FileNotFoundError(f"std.txt not found in {temp_path}")

        mean_cof = np.loadtxt(mean_file)
        std_cof = np.loadtxt(std_file)

        # Verify consistency (allowing for failed cycles)
        if len(cycle_data) != len(mean_cof):
            print(
                f"Warning: Cycle count mismatch for {temp}°C: "
                f"{len(cycle_data)} valid cycles vs {len(mean_cof)} in mean.txt"
            )

        return {
            'data': cycle_data,
            'mean_cof': mean_cof,
            'std_cof': std_cof,
            'n_cycles': len(cycle_data),
            'temp': temp,
            'failed_cycles': failed_cycles
        }

    def create_pytorch_dataset(
        self,
        temps: List[float] = [165, 167.5, 170]
    ) -> List[Dict]:
        """
        Create PyTorch-compatible dataset for all temperatures.

        Constructs feature tensors with 8 features per timestep:
        1. M_normalized: Transfer layer mass (placeholder, updated during training)
        2. T_normalized: Temperature normalized to [0, 1]
        3. sliding_distance_norm: Normalized position along slide
        4. velocity_x: Sliding velocity (normalized)
        5. force_x: Normal force x-component (normalized)
        6. force_y: Normal force y-component (normalized)
        7. force_z: Normal force z-component (normalized)
        8. cycle_phase: Phase within cycle [0, 1]

        Args:
            temps: List of temperatures to load

        Returns:
            List of dictionaries, one per cycle, with keys:
                - 'M': float, initial transfer layer mass (placeholder)
                - 'T': float, temperature (°C)
                - 'features': Tensor[n_timesteps, 8], input features
                - 'target_cof': Tensor[n_timesteps], observed COF
                - 'cycle_num': int, cycle number (1-indexed)
                - 'temp': float, temperature
                - 'mean_cof_observed': float, cycle-averaged COF from mean.txt
        """
        dataset = []

        # Normalization constants
        T_min, T_max = 25.0, 170.0
        velocity_scale = 0.1  # Typical velocity ~0.025 m/s
        force_scale = 10.0    # Typical force ~10 N

        for temp in temps:
            print(f"Loading data for {temp}°C...")
            temp_data = self.load_all_cycles_for_temperature(temp)

            T_norm = (temp - T_min) / (T_max - T_min)

            for cycle_idx, df in enumerate(temp_data['data']):
                n_timesteps = len(df)

                # Extract features (8 features per timestep)
                features = np.stack([
                    np.full(n_timesteps, 0.5),  # M placeholder (updated during training)
                    np.full(n_timesteps, T_norm),  # T normalized
                    df['sliding_distance_norm'].values,
                    df['velocity_x'].values / velocity_scale,
                    df['force_x'].values / force_scale,
                    df['force_y'].values / force_scale,
                    df['force_z'].values / force_scale,
                    df['cycle_phase'].values
                ], axis=1)  # Shape: [n_timesteps, 8]

                target_cof = df['cof'].values

                # Get corresponding mean COF from mean.txt
                # Account for possible cycle mismatches
                if cycle_idx < len(temp_data['mean_cof']):
                    mean_cof_observed = temp_data['mean_cof'][cycle_idx]
                else:
                    # Use actual mean if index out of bounds
                    mean_cof_observed = target_cof.mean()

                dataset.append({
                    'M': 0.5,  # Initial placeholder
                    'T': temp,
                    'features': torch.tensor(features, dtype=torch.float32),
                    'target_cof': torch.tensor(target_cof, dtype=torch.float32),
                    'cycle_num': cycle_idx + 1,
                    'temp': temp,
                    'mean_cof_observed': mean_cof_observed
                })

            print(f"  Loaded {len(temp_data['data'])} cycles")

        print(f"\nTotal dataset size: {len(dataset)} cycles")
        return dataset

    def get_dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """
        Compute statistics for the dataset.

        Args:
            dataset: PyTorch dataset from create_pytorch_dataset()

        Returns:
            Dictionary with statistics:
                - n_cycles_total: Total number of cycles
                - n_cycles_per_temp: Dict mapping temp -> count
                - avg_timesteps_per_cycle: Average points per cycle
                - total_timesteps: Total data points
                - cof_range: (min, max) COF values
                - force_range: Dict with x, y, z force ranges
        """
        n_cycles_total = len(dataset)

        n_cycles_per_temp = {}
        timesteps = []
        all_cof = []
        all_forces = {'x': [], 'y': [], 'z': []}

        for item in dataset:
            temp = item['temp']
            n_cycles_per_temp[temp] = n_cycles_per_temp.get(temp, 0) + 1

            timesteps.append(len(item['target_cof']))
            all_cof.extend(item['target_cof'].numpy())

            features = item['features'].numpy()
            all_forces['x'].extend(features[:, 4])
            all_forces['y'].extend(features[:, 5])
            all_forces['z'].extend(features[:, 6])

        return {
            'n_cycles_total': n_cycles_total,
            'n_cycles_per_temp': n_cycles_per_temp,
            'avg_timesteps_per_cycle': np.mean(timesteps),
            'total_timesteps': sum(timesteps),
            'cof_range': (min(all_cof), max(all_cof)),
            'force_range': {
                'x': (min(all_forces['x']), max(all_forces['x'])),
                'y': (min(all_forces['y']), max(all_forces['y'])),
                'z': (min(all_forces['z']), max(all_forces['z']))
            }
        }


def main():
    """Example usage and testing"""
    loader = HighFrequencyDataLoader()

    print("Testing data loader...")
    print("=" * 60)

    # Test loading single cycle
    print("\n1. Loading single cycle (165°C, cycle 1):")
    try:
        df = loader.load_cycle_csv(165, 1)
        print(f"   Raw data shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")

        df_clean = loader.preprocess_cycle(df)
        print(f"   After preprocessing: {df_clean.shape}")
        print(f"   Added columns: cycle_phase, sliding_distance_norm")
    except Exception as e:
        print(f"   Error: {e}")

    # Test loading all cycles for one temperature
    print("\n2. Loading all cycles for 165°C:")
    try:
        temp_data = loader.load_all_cycles_for_temperature(165)
        print(f"   Cycles loaded: {temp_data['n_cycles']}")
        print(f"   Mean COF length: {len(temp_data['mean_cof'])}")
        print(f"   Failed cycles: {len(temp_data['failed_cycles'])}")
    except Exception as e:
        print(f"   Error: {e}")

    # Create PyTorch dataset
    print("\n3. Creating PyTorch dataset:")
    try:
        dataset = loader.create_pytorch_dataset([165, 167.5, 170])

        stats = loader.get_dataset_statistics(dataset)
        print(f"\n   Dataset Statistics:")
        print(f"   - Total cycles: {stats['n_cycles_total']}")
        print(f"   - Cycles per temp: {stats['n_cycles_per_temp']}")
        print(f"   - Avg timesteps/cycle: {stats['avg_timesteps_per_cycle']:.1f}")
        print(f"   - Total timesteps: {stats['total_timesteps']}")
        print(f"   - COF range: [{stats['cof_range'][0]:.3f}, {stats['cof_range'][1]:.3f}]")

        print(f"\n   Sample data point (cycle 1):")
        print(f"   - Features shape: {dataset[0]['features'].shape}")
        print(f"   - Target COF shape: {dataset[0]['target_cof'].shape}")
        print(f"   - Temperature: {dataset[0]['T']}°C")
        print(f"   - Mean COF: {dataset[0]['mean_cof_observed']:.4f}")

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

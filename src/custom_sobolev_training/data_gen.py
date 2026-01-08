"""
Take in the Combined Dataset and create derivatives, Dataloaders, and others.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch


class DifferenceComputer(ABC):
    """
    Abstract base class for computing numerical derivatives using different finite difference schemes.
    
    Methods:
        compute_difference: Compute the derivative using the specific difference scheme. The inputs are
                            arrays of values and corresponding feature values for a single group. The outputs
                            are the computed derivatives and the indices of valid entries.
    """

    @abstractmethod
    def compute_difference(
        self, values: np.ndarray, feature: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the derivative using the specific difference scheme.

        Args:
            values: Array of label values for a single group (sorted by feature)
            feature: Array of feature values for a single group (sorted)

        Returns:
            Tuple of (derivatives, valid_indices) where valid_indices are the indices
            in the input arrays that have valid derivatives computed.
        """
        pass


class ForwardDifference(DifferenceComputer):
    """
    Compute derivatives using forward difference: f'(x) ≈ (f(x+h) - f(x)) / h
    """

    def compute_difference(
        self, values: np.ndarray, feature: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward difference derivatives.

        Args:
            values: Array of label values
            feature: Array of feature values

        Returns:
            Tuple of (derivatives, valid_indices)
        """
        if len(values) < 2:
            return np.array([]), np.array([])

        # Can only compute forward difference for all points except the last
        valid_indices = np.arange(len(values) - 1)

        # Compute differences
        delta_values = values[1:] - values[:-1]
        delta_feature = feature[1:] - feature[:-1]

        # Avoid division by zero - filter out entries with zero denominator
        mask = delta_feature != 0
        derivatives = delta_values[mask] / delta_feature[mask]
        valid_indices = valid_indices[mask]

        return derivatives, valid_indices


class BackwardDifference(DifferenceComputer):
    """
    Compute derivatives using backward difference: f'(x) ≈ (f(x) - f(x-h)) / h
    """

    def compute_difference(
        self, values: np.ndarray, feature: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute backward difference derivatives.

        Args:
            values: Array of label values
            feature: Array of feature values

        Returns:
            Tuple of (derivatives, valid_indices)
        """
        if len(values) < 2:
            return np.array([]), np.array([])

        # Can only compute backward difference for all points except the first
        valid_indices = np.arange(1, len(values))

        # Compute differences
        delta_values = values[1:] - values[:-1]
        delta_feature = feature[1:] - feature[:-1]

        # Avoid division by zero - filter out entries with zero denominator
        mask = delta_feature != 0
        derivatives = delta_values[mask] / delta_feature[mask]
        valid_indices = valid_indices[mask]

        return derivatives, valid_indices


class CentralDifference(DifferenceComputer):
    """
    Compute derivatives using central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    Allows skipping points on either side.
    """

    def __init__(self, skip: int = 1):
        """
        Initialize central difference computer.

        Args:
            skip: Number of points to skip on either side for computing the difference.
                  skip=1 means using immediate neighbors (x-h and x+h).
                  skip=2 means using (x-2h and x+2h), etc.
        """
        if skip < 1:
            raise ValueError("skip must be at least 1")
        self.skip = skip

    def compute_difference(
        self, values: np.ndarray, feature: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute central difference derivatives.

        Args:
            values: Array of label values
            feature: Array of feature values

        Returns:
            Tuple of (derivatives, valid_indices)
        """
        n = len(values)
        if n < 2 * self.skip + 1:
            return np.array([]), np.array([])

        # Can only compute central difference for points with enough neighbors
        valid_indices = np.arange(self.skip, n - self.skip)

        # Recompute more carefully
        delta_values = (
            values[valid_indices + self.skip] - values[valid_indices - self.skip]
        )
        delta_feature = (
            feature[valid_indices + self.skip] - feature[valid_indices - self.skip]
        )

        # Avoid division by zero - filter out entries with zero denominator
        mask = delta_feature != 0
        derivatives = delta_values[mask] / delta_feature[mask]
        valid_indices = valid_indices[mask]

        return derivatives, valid_indices


def generate_derivatives(
    data: np.ndarray,
    feature_indices: List[int],
    label_index: int,
    group_index: int,
    derivative_type: DifferenceComputer = ForwardDifference(),
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate derivatives of label with respect to features for grouped data.

    Args:
        data: 2D numpy array where rows are samples and columns are features/labels
        feature_indices: List of column indices that are features to compute derivatives for
        label_index: Column index of the label
        group_index: Column index to group by (e.g., sheep ID)
        derivative_type: Type of difference object to use - ForwardDifference, BackwardDifference, or CentralDifference

    Returns:
        Tuple of:
            - derivatives: 2D array of shape (n_valid_samples, n_features) containing derivatives
            - original_data: The original data rows that correspond to valid derivatives
            - kept_indices: List of original row indices that were kept

    Example:
        >>> data = np.array([[0, 1.0, 2.0, 5.0],
        ...                  [0, 1.5, 2.5, 6.0],
        ...                  [1, 2.0, 3.0, 7.0]])
        >>> derivatives, orig_data, indices = generate_derivatives(
        ...     data, feature_indices=[1, 2], label_index=3, group_index=0,
        ...     derivative_type='forward')
    """
    computer = derivative_type

    # Get unique groups
    groups = np.unique(data[:, group_index])

    all_derivatives = []
    all_original_data = []
    all_kept_indices = []

    for group_value in groups:
        # Get data for this group
        group_mask = data[:, group_index] == group_value
        group_data = data[group_mask]
        group_original_indices = np.where(group_mask)[0]

        # Get label values for this group
        label_values = group_data[:, label_index]

        # Compute derivatives for each feature
        group_derivatives = []
        valid_indices_per_feature = []

        for feature_idx in feature_indices:
            feature_values = group_data[:, feature_idx]
            label_values = group_data[:, label_index]

            # Compute derivatives
            derivatives, valid_indices = computer.compute_difference(
                label_values, feature_values
            )
            group_derivatives.append((derivatives, valid_indices))
            valid_indices_per_feature.append(set(valid_indices))

        # Find indices that are valid for all features (intersection)
        if valid_indices_per_feature:
            common_valid_indices = set.intersection(*valid_indices_per_feature)
            common_valid_indices = sorted(list(common_valid_indices))

            if len(common_valid_indices) > 0:
                # Extract derivatives for common valid indices
                derivatives_matrix = np.zeros(
                    (len(common_valid_indices), len(feature_indices))
                )

                for feat_num, (derivatives, valid_indices) in enumerate(
                    group_derivatives
                ):
                    # Create mapping from valid_indices to derivatives
                    idx_to_deriv = {
                        idx: deriv for idx, deriv in zip(valid_indices, derivatives)
                    }

                    # Fill in derivatives for common indices
                    for row_num, idx in enumerate(common_valid_indices):
                        derivatives_matrix[row_num, feat_num] = idx_to_deriv[idx]

                # Store results
                all_derivatives.append(derivatives_matrix)
                all_original_data.append(group_data[common_valid_indices])

                # Map back to global indices
                global_indices = group_original_indices[common_valid_indices]
                all_kept_indices.extend(global_indices)

    # Concatenate results from all groups
    if all_derivatives:
        derivatives = np.vstack(all_derivatives)
        original_data = np.vstack(all_original_data)
        kept_indices = all_kept_indices
    else:
        derivatives = np.array([]).reshape(0, len(feature_indices))
        original_data = np.array([]).reshape(0, data.shape[1])
        kept_indices = []

    return derivatives, original_data, kept_indices


def generate_dataloaders(
    dataset: np.ndarray,
    feature_indices: List[int],
    label_indices: List[int],
    group_index: int,
    validation_groups: List[int],
    batch_size: int,
    shuffle: bool = True,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Given a numpy dataset array, generate Train and Validation Dataloaders

    Args:
        dataset: 2D numpy array where rows are samples and columns are features/labels
        feature_indices: List of column indices to be used as features
        label_indices: List of column indices to be used as labels
        group_index: Column index to group by (e.g., sheep ID)
        validation_groups: List of group values to be used for validation set. The rest will be training set.
        batch_size: Batch size for the dataloaders
        shuffle: Whether to shuffle the training data
        device: Torch device to load the data onto
        **kwargs: Additional keyword arguments to pass directly to DataLoader

    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    # Split dataset into train and validation based on group
    val_mask = np.isin(dataset[:, group_index], validation_groups)
    train_data = dataset[~val_mask]
    val_data = dataset[val_mask]

    # Extract features and labels
    X_train = torch.tensor(train_data[:, feature_indices], dtype=torch.float32)
    X_train = X_train.to(device)
    y_train = torch.tensor(train_data[:, label_indices], dtype=torch.float32)
    y_train = y_train.to(device)
    X_val = torch.tensor(val_data[:, feature_indices], dtype=torch.float32)
    X_val = X_val.to(device)
    y_val = torch.tensor(val_data[:, label_indices], dtype=torch.float32)
    y_val = y_val.to(device)

    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    return train_loader, val_loader


if __name__ == "__main__":
    data = pd.read_csv("./data/combined_LLPSA2.csv")
    data["experiment_id"] = pd.factorize(data["experiment_id"])[0]
    data_np = data.to_numpy()
    features_idx = list(range(20))
    label_idx = 22
    group_idx = 20
    data_np[:, features_idx] = RobustScaler().fit_transform(data_np[:, features_idx])
    data_np[:, label_idx] = (
        RobustScaler().fit_transform(data_np[:, label_idx].reshape(-1, 1)).flatten()
    )
    derivatives, original_data, kept_indices = generate_derivatives(
        data_np,
        feature_indices=features_idx,
        label_index=label_idx,
        group_index=group_idx,
        derivative_type=CentralDifference(skip=1),
    )
    print("Derivatives shape:", derivatives.shape)
    print("Original data shape:", original_data.shape)
    print("Data Points Dropped:", data_np.shape[0] - original_data.shape[0])
    derivative_df = pd.DataFrame(
        derivatives, columns=[f"dlabel_dfeat_{i}" for i in features_idx]
    )
    print("Hold")


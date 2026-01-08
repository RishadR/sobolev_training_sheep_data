"""
Code to reject outliers using different schemes
"""

from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np


class OutlierRejector(ABC):
    """
    Base class for outlier rejection schemes

    Methods
    -------
    reject_outliers(dataset, column_indices)
        Reject outliers in the specified columns of the dataset. Returns the modified dataset and the list of indices
        of rows that were kept (i.e., not rejected as outliers).

    """

    @abstractmethod
    def reject_outliers(
        self, dataset: np.ndarray, column_indices: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        pass


class StdDevOutlierRejector(OutlierRejector):
    """
    Reject outliers that are beyond m standard deviations from the mean across all specified columns.
    """

    def __init__(self, m: float = 3.0):
        self.m = m

    def reject_outliers(
        self, dataset: np.ndarray, column_indices: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        mean = np.mean(dataset[:, column_indices], axis=0)
        std_dev = np.std(dataset[:, column_indices], axis=0)
        mask = np.all(
            np.abs(dataset[:, column_indices] - mean) <= self.m * std_dev, axis=1
        )
        kept_indices = np.where(mask)[0].tolist()
        return dataset[mask], kept_indices

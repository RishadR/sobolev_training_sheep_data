"""
More complex and interesting finite difference schemes
"""
from typing import Tuple
import numpy as np
from custom_sobolev_training.data_gen import DifferenceComputer
# from tfo_sim2.four_layer_model_optical_props_table import get_blood_filled_tissue_mu_a

class LabelBinnedDifference(DifferenceComputer):
    """
    Bins the label values into small, discrete bins and computes finite differences between bins. The differences are
    computed using the average label values and average feature values within each bin. This averaged difference is 
    assigned to all data points within the bin. Computes a forward differece and the very last bin is ignored.
    """
    def __init__(self, bin_width: float):
        self.bin_width = bin_width
    
    def compute_difference(self, values: np.ndarray, feature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        min_label = np.min(values)
        max_label = np.max(values)
        bins = np.arange(min_label, max_label + self.bin_width, self.bin_width)
        bin_indices = np.digitize(values, bins) - 1  # Get bin indices for each value

        derivative = np.full_like(values, np.nan)
        valid_indices = []

        for b in range(len(bins) - 1):
            bin_mask = (bin_indices == b)
            if np.sum(bin_mask) == 0:
                continue
            
            next_bin_mask = (bin_indices == b + 1)
            if np.sum(next_bin_mask) == 0:
                continue
            
            avg_value_current = np.mean(values[bin_mask])
            avg_feature_current = np.mean(feature[bin_mask])
            avg_value_next = np.mean(values[next_bin_mask])
            avg_feature_next = np.mean(feature[next_bin_mask])

            diff_value = avg_value_next - avg_value_current
            diff_feature = avg_feature_next - avg_feature_current

            if diff_feature != 0:
                bin_derivative = diff_value / diff_feature
                derivative[bin_mask] = bin_derivative
                valid_indices.extend(np.where(bin_mask)[0])
        derivative = derivative[valid_indices]
        return derivative, np.array(valid_indices)

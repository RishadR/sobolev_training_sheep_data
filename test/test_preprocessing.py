"""
Unit tests for preprocessing.py
"""

import unittest
from unittest import TestCase
import numpy as np
from custom_sobolev_training.data_generation.preprocessing import compute_central_difference


class TestPreprocessing(TestCase):
    def test_compute_central_difference(self):
        data_series = np.array([1, 2, 1, 1, 1])
        expected = np.array([-1. ,  0. ,  0.5,  0. ,  0.5])
        result = compute_central_difference(data_series)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()

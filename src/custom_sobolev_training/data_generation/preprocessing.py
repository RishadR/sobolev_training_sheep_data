"""
Experimental data preparation for sheep data. 
Only need to run this script once to generate the processed data.

(Credits to Weitai Qian for AC/DC Extraction Code).
"""

import math
from typing import Literal
import numpy as np
import emd
from .signal_api import moving_average, butter_lowpass_filter, butter_bandpass_filter, lockin_separation


def extract_DC(ppg, Fs: float = 80.0, method: Literal["envelope", "lp_filter"] = "envelope", smooth: bool = True):
    """
    Extract the DC component of the PPG signal.

    Parameters
    ----------
    - ppg: 2D array of PPG signals. Each column represents a channel.
    - Fs: The sampling frequency.
    - method: The method to use for extracting the DC component. Options: 'envelope' or 'lp_filter'.
    - smooth: Whether to smooth the DC component.

    Returns
    -------
    - The DC component of the PPG signal, as a 2D array, where each column represents a channel.
    """
    ppg_dc = np.zeros_like(ppg)
    for i in range(ppg.shape[1]):
        if method == "envelope":
            ppg_dc[:, i] = emd.sift.interp_envelope(ppg[:, i], mode="lower", interp_method="pchip")
        elif method == "lp_filter":
            ppg_dc[:, i] = butter_lowpass_filter(ppg[:, i], 0.1, fs=Fs, order=10)
        else:
            raise ValueError(f"Unknown method {method}.")
        if smooth:
            # Moving average window size: 60s
            ppg_dc[:, i] = moving_average(ppg_dc[:, i], 4800)
    return ppg_dc


def extract_AC(ppg: np.ndarray, freq, Fs=80, low_cut=None, high_cut=None, smooth=True) -> np.ndarray:
    """
    Extract the AC component of the PPG signal.

    Parameters
    ----------
    - ppg: 2D array of PPG signals. Each column represents a channel.
    - freq: The frequency of the AC component - in Hz.
    - Fs: The sampling frequency.
    - low_cut: The low cut-off frequency for the bandpass filter.
    - high_cut: The high cut-off frequency for the bandpass filter.
    - smooth: Whether to smooth the AC component.

    Returns
    -------
    - The AC component of the PPG signal, as a 2D array, where each column represents a channel.
    """
    ppg_ac_component = np.zeros_like(ppg)
    for i in range(ppg.shape[1]):
        ppg_channel = ppg[:, i]
        if low_cut is not None and high_cut is not None:
            ppg_channel = butter_bandpass_filter(ppg_channel, low_cut, high_cut, fs=Fs, order=5)
        lockin_ac = lockin_separation(np.repeat(freq, 80), ppg_channel, Fs=80, cut_off_freq=0.1)
        if smooth:
            lockin_ac = moving_average(lockin_ac, 4800)
        ppg_ac_component[:, i] = lockin_ac
    return ppg_ac_component


def average_chunks(input_signal, win_len=1.5):
    """
    Averages each win_len seconds of the signal(along axis 0) into a single value.
    """
    pad_width = int((win_len * 80 * 60) // 2)
    padded_signal = []
    num_ch = input_signal.shape[1]
    for i in range(num_ch):
        padded_signal.append(np.pad(input_signal[:, i], pad_width=pad_width, mode="edge"))
    padded_signal = np.column_stack(padded_signal)
    ratio_avg = []
    for i in range(len(input_signal) // 80):
        start_idx = i * 80
        end_idx = i * 80 + 2 * pad_width
        input_signal = np.mean(padded_signal[start_idx:end_idx], axis=0)
        ratio_avg.append(input_signal)
    return np.array(ratio_avg)


def round_to_decimals(number, decimal_places, method="round"):
    """
    Round a number to the specified number of decimal places using the specified rounding method.

    Parameters
    ----------
    - number: The number to be rounded.
    - decimal_places: The number of decimal places to round to.
    - method: The rounding method to use. Default is 'round'.
              Options: 'ceil' - round up to the nearest decimal place.
                       'floor' - round down to the nearest decimal place.
                       'round' - round to the nearest decimal place (default).

    Returns
    -------
    - The rounded number.
    """
    multiplier = 10**decimal_places
    if method == "ceil":
        return math.ceil(number * multiplier) / multiplier
    elif method == "floor":
        return math.floor(number * multiplier) / multiplier
    elif method == "round":
        return round(number * multiplier) / multiplier
    else:
        raise ValueError("Invalid method. Use 'ceil', 'floor', or 'round'.")


def reject_outliers(data_series: np.ndarray, m=2) -> np.ndarray:
    """
    Reject outliers in the data beyond m-th standard deviation from the mean.

    Parameters
    ----------
        - data: The data to be processed. The time series should extend along each column.
        - m: The number of standard deviations from the mean to consider as outliers.

    Returns
    -------
        - The data with outliers replaced with NaN.
    """
    mean = np.mean(data_series, axis=0)
    std_dev = np.std(data_series, axis=0)
    outliers = np.abs(data_series - mean) > m * std_dev
    data_series[outliers] = np.nan
    return data_series


def impute_with_left(data_series: np.ndarray) -> np.ndarray:
    """
    Convert NaN values in the data to the last known value.

    Parameters
    ----------
        - data_series: The data to be processed. The time series should extend along each column.

    Returns
    -------
        - The imputed data.
    """
    ## If the first value is NaN, replace it with the first non-NaN value
    for i in range(data_series.shape[1]):
        if np.isnan(data_series[0, i]):
            first_non_nan = np.nan
            for j in range(data_series.shape[0]):
                if not np.isnan(data_series[j, i]):
                    first_non_nan = data_series[j, i]
                    break
            data_series[:j, i] = first_non_nan

    ## Replace NaN values with the last known value
    for i in range(data_series.shape[1]):
        for j in range(1, data_series.shape[0]):
            if np.isnan(data_series[j, i]):
                data_series[j, i] = data_series[j - 1, i]
    return data_series


def compute_central_difference(data_series: np.ndarray) -> np.ndarray:
    """
    Compute the central difference for each element in the data series.

    Parameters
    ----------
    - data_series: The data to be processed. The time series should extend along each column.

    Returns
    -------
    - The central difference of the data series. (Same size as input, 1D array)

    Notes:
    - The central difference is calculated using the formula:
      central_diff[i] = (data_series[i+1] - data_series[i-1]) / 2
    - The first and last elements are calculated using the forward and backward differences respectively. Might want to
        drop them if they are not needed.
    """
    data_series = data_series.flatten()
    kernel = np.array([-0.5, 0, 0.5])
    central_diff = np.convolve(data_series, kernel, mode="same")
    return central_diff

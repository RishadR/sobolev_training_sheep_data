"""
Time series signal processing functions.
"""

import numpy as np
import math
from scipy.signal import argrelextrema, butter, filtfilt, sosfiltfilt


def moving_average(signal, window_size):
    """
    Calculate the moving average of a given signal over a specified window size.

    Args:
    signal (array_like): Input array or object that can be converted to an array.
    window_size (int): The size of the window to take the average over.

    Returns:
    array: A smoothed array with the same shape as `signal`.
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(signal, window, "same")


def lower_envelope(signal):
    """
    Calculate the lower envelope of a given signal.

    Args:
    signal (array_like): Input array or object that can be converted to an array.

    Returns:
    array: An array representing the lower envelope of the input signal.
    """
    # Find the local minima indices
    minima_indices = argrelextrema(signal, np.less)[0]

    # Extract the values at the minima indices
    envelope_values = signal[minima_indices]

    # Create a piecewise linear function using the minima indices and values
    envelope_function = np.poly1d(np.polyfit(minima_indices, envelope_values, deg=1))

    # Evaluate the function over the entire range of the signal
    envelope = envelope_function(np.arange(len(signal)))

    return envelope


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="lowpass")
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="highpass")
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def myfilter_sos(inp, Wn, N, filter_type, plot=False):
    # filter_type = 'bandpass', 'high', 'low'

    sos = butter(N, Wn, filter_type, fs=80, output="sos")

    if len(inp.shape) > 1:
        out_sig = np.empty_like(inp)
        for ch in range(inp.shape[1]):
            out_sig[:, ch] = sosfiltfilt(sos, inp[:, ch])
        return out_sig
    return sosfiltfilt(sos, inp)


def lockin_separation(HR, sig_TFO, Fs, cut_off_freq=0.2):
    I = 2 * np.sin(np.cumsum(HR) / Fs * 2 * math.pi)
    Q = 2 * np.cos(np.cumsum(HR) / Fs * 2 * math.pi)

    I = np.multiply(sig_TFO[: len(HR)], I[: len(sig_TFO)])
    Q = np.multiply(sig_TFO[: len(HR)], Q[: len(sig_TFO)])
    order = 5  # filter order
    I = myfilter_sos(I, cut_off_freq, order, "low")
    Q = myfilter_sos(Q, cut_off_freq, order, "low")
    return np.sqrt(np.multiply(I, I) + np.multiply(Q, Q))

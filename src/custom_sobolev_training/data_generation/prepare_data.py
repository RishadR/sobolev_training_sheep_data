"""
Run this script to prepare the data for training the Sobolev Model
"""

from pathlib import Path
import numpy as np
import pandas as pd
from .preprocessing import extract_AC, extract_DC, impute_with_left, round_to_decimals, reject_outliers, average_chunks


if __name__ == "__main__":
    base_data_path = Path(__file__).parent.parent.parent.parent.resolve() / "data"
    ppg_base_path = base_data_path / "ppg"
    available_datasets = [
        ppg_base_path / "sheepAround1.csv",
        ppg_base_path / "sheepAround2.csv",
        ppg_base_path / "sheepAround3.csv",
    ]

    SAMPLING_FREQ = 80

    ## Process each dataset and save the processed data
    for ppg_file in available_datasets:
        ## Load Data
        fhr_data_path = base_data_path / "fhr" / ppg_file.name
        fSaO2_data_path = base_data_path / "fSaO2" / ppg_file.name

        data = pd.read_csv(ppg_file)  # 80 Samples per second
        fhr_data = pd.read_csv(fhr_data_path)  # 1 Sample per second
        fSaO2_data = pd.read_csv(fSaO2_data_path)  # 1 Sample per second

        data.columns = data.columns.sort_values()  # ch1wv1, ch1wv2, ch1wv3, ch2wv1, ch2wv2, ...
        data: np.ndarray = data.to_numpy()
        data = impute_with_left(data)  # Impute Missing Values in case of NaN
        fhr_series: np.ndarray = fhr_data.iloc[:, 0].dropna().to_numpy()
        fhr_series /= 60  # convert bpm values to Hz values
        fSaO2_series: np.ndarray = fSaO2_data.iloc[:, 0].dropna().to_numpy()

        ## Truncate Data to be the same length
        data_len = min(len(data) // SAMPLING_FREQ, len(fhr_series), len(fSaO2_series))
        data = data[: data_len * SAMPLING_FREQ, :]
        fhr_series = fhr_series[:data_len]
        fSaO2_series = fSaO2_series[:data_len]

        ## Extract AC and DC components
        bp_low_cutoff = round_to_decimals(np.min(fhr_series), 1, "floor")
        bp_high_cutoff = round_to_decimals(np.max(fhr_series), 1, "ceil")
        ac_component = extract_AC(data, fhr_series, SAMPLING_FREQ, bp_low_cutoff, bp_high_cutoff, True)
        dc_component = extract_DC(data, SAMPLING_FREQ, "envelope", True)

        ## Crop first and last 5 seconds of data to remove transient effects
        LEFT_CROP_SEC = 5
        RIGHT_CROP_SEC = 5
        ac_component = ac_component[LEFT_CROP_SEC * SAMPLING_FREQ : -RIGHT_CROP_SEC * SAMPLING_FREQ, :]
        dc_component = dc_component[LEFT_CROP_SEC * SAMPLING_FREQ : -RIGHT_CROP_SEC * SAMPLING_FREQ, :]
        fSaO2_series = np.repeat(fSaO2_series, SAMPLING_FREQ).reshape(-1, 1)  # Convert from 1 Hz to 80 Hz Sampling
        fSaO2_series = fSaO2_series[LEFT_CROP_SEC * SAMPLING_FREQ : -RIGHT_CROP_SEC * SAMPLING_FREQ]

        ## Calculate Ratio per wavelength
        ratio = (2 * ac_component + dc_component) / dc_component

        ## Outlier Rejection & Impute Missing Values
        ratio = reject_outliers(ratio)
        ratio = impute_with_left(ratio)

        ## Average Ratio for each 1 second chunk
        ratio = average_chunks(ratio)
        fSaO2_series = average_chunks(fSaO2_series)

        ## Save the processed data
        channel_names = []
        save_path = base_data_path / "processed" / ppg_file.name
        for channel_name in [f"ch{i+1}" for i in range(5)]:
            for wavelength in [f"wv{i+1}" for i in range(3)]:
                channel_names.append(f"ratio_{channel_name}_{wavelength}")
        channel_names.append("fSaO2")
        processed_data = np.column_stack([ratio, fSaO2_series])
        processed_df = pd.DataFrame(processed_data, columns=channel_names)
        processed_df.to_csv(save_path, index=False)

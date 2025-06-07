# --- START OF FILE utils.py ---
import numpy as np
import pandas as pd # For smoothing in preprocess, if used
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis # For skewness and kurtosis
import os

def load_sensor_data_original(filepath, expected_cols=6):
    """Loads original sensor data from a single txt file."""
    try:
        data = np.loadtxt(filepath, dtype=np.float32)
        if data.ndim == 1:
            if len(data) == 0: # Handle empty file or line
                print(f"Warning: File {filepath} is empty or contains an empty line. Returning None.")
                return None
            data = data.reshape(1, -1)
        if data.shape[1] != expected_cols:
            print(f"Warning: File {filepath} has {data.shape[1]} columns, expected {expected_cols}. Returning None.")
            return None
        if data.shape[0] == 0: # Handle case where file has columns but no rows
            print(f"Warning: File {filepath} has no data rows. Returning None.")
            return None
        return data
    except ValueError as ve: # Specifically catch errors from np.loadtxt (e.g. non-numeric data)
        print(f"Error loading {filepath} due to ValueError (likely non-numeric data): {ve}. Returning None.")
        return None
    except Exception as e:
        print(f"Generic error loading {filepath}: {e}. Returning None.")
        return None

def detrend(signal_column):
    """Removes linear trend from a 1D signal (a single column)."""
    signal_1d = signal_column.squeeze() # Make sure it's 1D
    if signal_1d.ndim == 0: # Handle scalar input after squeeze
        return signal_column
    if len(signal_1d) < 2:
        return signal_column.reshape(-1,1) if signal_column.ndim == 1 else signal_column

    time_axis = np.arange(len(signal_1d))
    try:
        p = np.polyfit(time_axis, signal_1d, 1)
        trend = np.polyval(p, time_axis)
        detrended_signal = signal_1d - trend
        return detrended_signal.reshape(-1, 1)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Warning: Could not detrend signal (length {len(signal_1d)}) due to: {e}. Returning original.")
        return signal_column.reshape(-1,1) if signal_column.ndim == 1 else signal_column


def create_new_feature_and_detrend(original_sequence_6col):
    """Calculates acc_sq_sum and detrends it."""
    if original_sequence_6col is None or original_sequence_6col.shape[1] != 6 or original_sequence_6col.shape[0] == 0:
        return None, None
    ax = original_sequence_6col[:, 0]
    ay = original_sequence_6col[:, 1]
    az = original_sequence_6col[:, 2]
    acc_sq_sum = ax**2 + ay**2 + az**2
    acc_sq_sum_col = acc_sq_sum.reshape(-1, 1)

    detrended_acc_sq_sum_col = detrend(acc_sq_sum_col) # detrend now handles short signals
    
    return acc_sq_sum_col, detrended_acc_sq_sum_col

def get_peak_signal(original_sequence_6col, smoothing_window=None):
    """
    Prepares the signal used for peak detection.
    Returns the (optionally smoothed) detrended acc_sq_sum_col.
    """
    _, detrended_feature = create_new_feature_and_detrend(original_sequence_6col)
    if detrended_feature is not None:
        signal = detrended_feature.squeeze()
        if smoothing_window and len(signal) >= smoothing_window:
            signal = pd.Series(signal).rolling(window=smoothing_window, center=True, min_periods=1).mean().to_numpy()
        return signal
    return None

def segment_around_peaks(data_7col, peak_indices, window_before, window_after, min_valid_ratio=0.3):
    """
    Segments data around detected peaks.
    If a segment has less than min_valid_ratio of actual data (rest is padding), it's skipped.
    """
    segments = []
    total_window_length = window_before + 1 + window_after
    if data_7col.shape[1] == 0: # Should not happen if data_7col is properly formed
        return segments
    num_channels = data_7col.shape[1]

    for peak_idx in peak_indices:
        start_idx = peak_idx - window_before
        end_idx = peak_idx + window_after + 1

        segment = np.zeros((total_window_length, num_channels), dtype=np.float32)

        src_start = max(0, start_idx)
        src_end = min(data_7col.shape[0], end_idx)

        valid_data_length_in_segment = src_end - src_start
        
        if valid_data_length_in_segment <= 0: # No overlap
            continue

        valid_ratio = valid_data_length_in_segment / total_window_length
        if valid_ratio < min_valid_ratio:
            # print(f"Segment around peak {peak_idx} skipped, valid data ratio {valid_ratio:.2f} < {min_valid_ratio}")
            continue

        dest_start = max(0, -start_idx) # If start_idx is negative, data starts at dest_start in segment
        dest_end = dest_start + valid_data_length_in_segment
        
        segment[dest_start:dest_end, :] = data_7col[src_start:src_end, :]
        segments.append(segment)
    return segments

def pad_or_truncate(sequence, target_length, num_channels):
    """Pads with zeros or truncates sequence to target_length for a given number of channels."""
    current_length = sequence.shape[0]
    if current_length == target_length:
        return sequence
    elif current_length > target_length:
        return sequence[:target_length, :]
    else:
        padding_size = target_length - current_length
        # Pad after the sequence data ((0, padding_size) for axis 0, (0,0) for axis 1)
        return np.pad(sequence, ((0, padding_size), (0, 0)), 'constant', constant_values=0)

def adjust_sum_to_one(probabilities, decimal_places):
    """
    Adjusts a list/array of probabilities so that they sum to 1.0,
    after an initial rounding to `decimal_places`.
    The adjustment is added to the element with the largest probability.
    """
    rounded_probs = np.round(np.array(probabilities, dtype=float), decimal_places) # Ensure float for operations

    current_sum = np.sum(rounded_probs)
    difference = 1.0 - current_sum

    # Only adjust if the difference is significant enough (more than typical float precision errors)
    if not np.isclose(difference, 0.0, atol=1e-9): # Use a small tolerance
        if len(rounded_probs) > 0:
            # Add difference to the largest element to minimize relative change
            # If all are zero (e.g. from a bad prediction), this won't work well, but sum is already 0.
            # If sum is already > 0, then at least one element is > 0.
            if np.any(rounded_probs): # Check if any element is non-zero
                 idx_to_adjust = np.argmax(rounded_probs)
            else: # if all are zero, and sum is 0, difference is 1. Distribute evenly or pick first.
                 idx_to_adjust = 0

            rounded_probs[idx_to_adjust] += difference
            # Clip to ensure probabilities remain in [0, 1] after adjustment
            rounded_probs[idx_to_adjust] = np.clip(rounded_probs[idx_to_adjust], 0.0, 1.0)
            # Re-round the adjusted element
            rounded_probs[idx_to_adjust] = np.round(rounded_probs[idx_to_adjust], decimal_places)

    # Final normalization pass to ensure sum is exactly 1, then re-round.
    # This handles any remaining small discrepancies from the above adjustment and rounding.
    sum_after_adjust = np.sum(rounded_probs)
    if sum_after_adjust > 1e-9 and not np.isclose(sum_after_adjust, 1.0, atol=1e-9):
        rounded_probs = rounded_probs / sum_after_adjust
        rounded_probs = np.round(rounded_probs, decimal_places) # Re-round after division

        # One last micro-adjustment if the sum is *still* off due to the last rounding
        final_final_sum = np.sum(rounded_probs)
        final_final_diff = 1.0 - final_final_sum
        if not np.isclose(final_final_diff, 0.0, atol=1e-9) and len(rounded_probs) > 0:
            if np.any(rounded_probs): idx_to_adjust_ff = np.argmax(rounded_probs)
            else: idx_to_adjust_ff = 0
            rounded_probs[idx_to_adjust_ff] += final_final_diff
            # Do not re-round this last micro-adjustment to preserve the sum=1 as much as possible
            # but ensure it's still within bounds and at correct decimal places for the element itself.
            rounded_probs[idx_to_adjust_ff] = np.round(np.clip(rounded_probs[idx_to_adjust_ff],0.0,1.0), decimal_places)


    # Ensure no negative probabilities due to aggressive subtraction of a large positive difference
    rounded_probs = np.clip(rounded_probs, 0.0, 1.0)
    return rounded_probs

def extract_statistical_features(sequence_data_segment):
    """
    Extracts statistical features from a single segment of time series data.
    Args:
        sequence_data_segment (np.array): Shape (sequence_length, num_channels)
    Returns:
        np.array: A 1D array of extracted features.
    """
    features = []
    num_channels = sequence_data_segment.shape[1]
    for i in range(num_channels):
        channel_data = sequence_data_segment[:, i]
        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))
        features.append(np.min(channel_data))
        features.append(np.max(channel_data))
        features.append(np.median(channel_data))
        features.append(skew(channel_data))
        features.append(kurtosis(channel_data))
        features.append(np.sum(channel_data**2)) # Energy
        zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
        features.append(zero_crossings)
        # Add more features as needed: FFT coefficients, wavelet coefficients, autocorrelation, etc.
    return np.array(features)
# --- END OF FILE utils.py ---
# --- START OF FILE preprocess_data.py ---
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
from tqdm import tqdm
from utils import (
    load_sensor_data_original,
    create_new_feature_and_detrend,
    get_peak_signal,
    segment_around_peaks
)

# --- Configuration for Preprocessing ---
# Source Data
ORIGINAL_BASE_DATA_DIR_TRAIN = "39_Training_Dataset"
ORIGINAL_TRAIN_INFO_FILE = os.path.join(ORIGINAL_BASE_DATA_DIR_TRAIN, "train_info.csv")
ORIGINAL_VALID_INFO_FILE = os.path.join(ORIGINAL_BASE_DATA_DIR_TRAIN, "valid_info.csv")
ORIGINAL_TRAIN_DATA_FOLDER = os.path.join(ORIGINAL_BASE_DATA_DIR_TRAIN, "train_data")

ORIGINAL_BASE_DATA_DIR_TEST = "39_Test_Dataset"
ORIGINAL_TEST_INFO_FILE = os.path.join(ORIGINAL_BASE_DATA_DIR_TEST, "test_info.csv")
ORIGINAL_TEST_DATA_FOLDER = os.path.join(ORIGINAL_BASE_DATA_DIR_TEST, "test_data")

# Segmented Data Output
SEGMENTED_DATA_BASE_DIR = "segmented_data_final" # CHANGED for new run
SEGMENTED_TRAIN_DATA_DIR = os.path.join(SEGMENTED_DATA_BASE_DIR, "train")
SEGMENTED_VALID_DATA_DIR = os.path.join(SEGMENTED_DATA_BASE_DIR, "valid")
SEGMENTED_TEST_DATA_DIR = os.path.join(SEGMENTED_DATA_BASE_DIR, "test")

SEGMENTED_TRAIN_INFO_FILE = os.path.join(SEGMENTED_DATA_BASE_DIR, "train_info_segmented.csv")
SEGMENTED_VALID_INFO_FILE = os.path.join(SEGMENTED_DATA_BASE_DIR, "valid_info_segmented.csv")
SEGMENTED_TEST_INFO_FILE = os.path.join(SEGMENTED_DATA_BASE_DIR, "test_info_segmented.csv")

# Segmentation Parameters - CRITICAL TUNING REQUIRED
WINDOW_BEFORE_PEAK = 29
WINDOW_AFTER_PEAK = 30
SEQUENCE_LENGTH = WINDOW_BEFORE_PEAK + 1 + WINDOW_AFTER_PEAK # Total 120
MIN_ORIGINAL_LENGTH_THRESHOLD = SEQUENCE_LENGTH // 2 # Example: Original must be at least half the segment length

# Peak Detection Parameters - CRITICAL TUNING REQUIRED
# Visualize your 'peak_detection_signal' to set these effectively!
PEAK_SIGNAL_SMOOTHING_WINDOW = 5 # e.g., 3, 5, 7. Use None to disable.
MIN_PEAK_HEIGHT = None          # Often better to use prominence.
MIN_PEAK_PROMINENCE = 0.2       # Example: Peak must be 0.2 units more prominent than surroundings.
MIN_PEAK_DISTANCE = SEQUENCE_LENGTH // 3 # Min distance between peaks.
MAX_PEAKS_PER_FILE = 30         # Max segments per original file.
MIN_SEGMENT_VALID_RATIO = 0.5   # Segment must have at least 50% real data.

# --- Main Preprocessing Function ---
def preprocess_and_segment_data(info_file_original, original_data_folder,
                                segmented_data_dir, segmented_info_file_out,
                                is_test_set=False):
    print(f"Processing {info_file_original}...")
    try:
        original_df = pd.read_csv(info_file_original, dtype={'unique_id': str})
        original_df.dropna(subset=['unique_id'], inplace=True)
        original_df['unique_id'] = original_df['unique_id'].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and '.' in str(x) and str(x).replace('.', '', 1).isdigit() else str(x)
        )
        original_df = original_df[original_df['unique_id'].str.lower() != 'nan']
    except FileNotFoundError:
        print(f"Error: Original info file not found: {info_file_original}")
        return
    if original_df.empty:
        print(f"Warning: Original info file {info_file_original} is empty after cleaning.")
        return

    os.makedirs(segmented_data_dir, exist_ok=True)
    all_segmented_info = []
    total_segments_before_filter = 0
    files_with_no_peaks_at_all = 0
    files_with_no_peaks_after_height_filter = 0

    for index, row in tqdm(original_df.iterrows(), total=original_df.shape[0], desc=f"Segmenting {os.path.basename(info_file_original)}"):
        original_unique_id = str(row['unique_id'])
        filepath_original = os.path.join(original_data_folder, f"{original_unique_id}.txt")

        original_sequence_6col = load_sensor_data_original(filepath_original)

        if original_sequence_6col is None or original_sequence_6col.shape[0] < MIN_ORIGINAL_LENGTH_THRESHOLD:
            continue

        peak_detection_signal = get_peak_signal(original_sequence_6col, smoothing_window=PEAK_SIGNAL_SMOOTHING_WINDOW)
        if peak_detection_signal is None or len(peak_detection_signal) < MIN_PEAK_DISTANCE:
            continue

        # 1. 第一次峰值檢測 (較寬鬆，用於計算平均峰高)
        #    使用 prominence 和 distance 來初步篩選，不使用 height
        candidate_peak_indices, candidate_properties = find_peaks(peak_detection_signal,
                                                                distance=MIN_PEAK_DISTANCE,
                                                                prominence=MIN_PEAK_PROMINENCE) # 初始較寬鬆的 prominence

        if len(candidate_peak_indices) == 0:
            files_with_no_peaks_at_all +=1
            # print(f"No initial candidate peaks for {original_unique_id}.")
            continue

        # 2. 計算動態 MIN_PEAK_HEIGHT
        #    candidate_properties['peak_heights'] 包含了這些候選峰值的高度
        #    如果 find_peaks 由於 distance/prominence 等原因沒有返回 'peak_heights' (應該會返回)
        #    則直接取 peak_detection_signal 在 candidate_peak_indices 處的值
        if 'peak_heights' in candidate_properties and len(candidate_properties['peak_heights']) > 0:
            avg_peak_height = np.mean(candidate_properties['peak_heights'])
        elif len(candidate_peak_indices) > 0 : # Fallback if 'peak_heights' not in properties
            avg_peak_height = np.mean(peak_detection_signal[candidate_peak_indices])
        else: # Should not happen if previous check passed
            files_with_no_peaks_at_all +=1
            continue


        dynamic_min_peak_height = avg_peak_height * 0.5 # 平均峰值的一半

        # 3. 第二次峰值檢測或篩選 (使用動態 MIN_PEAK_HEIGHT)
        #    可以直接在第一次的結果上篩選，或者重新調用 find_peaks 加上 height 參數
        #    選項 A: 在第一次結果上篩選 (更高效)
        final_peak_indices_after_height_filter = []
        final_properties_for_sorting = {'prominences': [], 'peak_heights': []} # For consistent sorting later

        for i, p_idx in enumerate(candidate_peak_indices):
            current_peak_height = candidate_properties['peak_heights'][i] if 'peak_heights' in candidate_properties else peak_detection_signal[p_idx]
            current_prominence = candidate_properties['prominences'][i] if 'prominences' in candidate_properties else 0 # Default if not calculated

            if current_peak_height >= dynamic_min_peak_height:
                final_peak_indices_after_height_filter.append(p_idx)
                # Store properties for these valid peaks for later sorting
                if 'prominences' in candidate_properties:
                    final_properties_for_sorting['prominences'].append(current_prominence)
                if 'peak_heights' in candidate_properties: # Should exist
                    final_properties_for_sorting['peak_heights'].append(current_peak_height)


        final_peak_indices = np.array(final_peak_indices_after_height_filter)
        
        # Convert lists in final_properties_for_sorting to numpy arrays
        if final_properties_for_sorting['prominences']:
            final_properties_for_sorting['prominences'] = np.array(final_properties_for_sorting['prominences'])
        if final_properties_for_sorting['peak_heights']:
            final_properties_for_sorting['peak_heights'] = np.array(final_properties_for_sorting['peak_heights'])


        if len(final_peak_indices) == 0:
            files_with_no_peaks_after_height_filter +=1
            # print(f"No peaks remaining after dynamic height filter for {original_unique_id} (dynamic_min_height: {dynamic_min_peak_height:.2f}).")
            continue

        # 4. 如果峰值仍然過多，則按突出度或高度排序選擇前 N 個
        if len(final_peak_indices) > MAX_PEAKS_PER_FILE:
            if len(final_properties_for_sorting['prominences']) == len(final_peak_indices) and np.any(final_properties_for_sorting['prominences']): # Check if prominences were stored and are not all zero
                sorted_indices = np.argsort(final_properties_for_sorting['prominences'])[::-1]
            elif len(final_properties_for_sorting['peak_heights']) == len(final_peak_indices):
                sorted_indices = np.argsort(final_properties_for_sorting['peak_heights'])[::-1]
            else: # Fallback: no valid properties to sort by, take first N (less ideal)
                # print(f"Warning: Too many peaks for {original_unique_id}, but no valid prominence/height for sorting. Taking first {MAX_PEAKS_PER_FILE}.")
                sorted_indices = np.arange(len(final_peak_indices)) # Or could sort final_peak_indices by their values in peak_detection_signal

            top_n_indices_in_final_peaks = sorted_indices[:MAX_PEAKS_PER_FILE]
            selected_peak_indices = final_peak_indices[top_n_indices_in_final_peaks]
            final_peak_indices = np.sort(selected_peak_indices) # 保持時間順序

        if len(final_peak_indices) == 0: # Should not happen if previous checks passed and MAX_PEAKS_PER_FILE > 0
            continue


        # 5. 準備7通道數據並分割
        _, detrended_7th_feature = create_new_feature_and_detrend(original_sequence_6col)
        if detrended_7th_feature is None: continue
        data_to_segment_7col = np.concatenate((original_sequence_6col, detrended_7th_feature), axis=1)

        segments_7col = segment_around_peaks(data_to_segment_7col, final_peak_indices,
                                             WINDOW_BEFORE_PEAK, WINDOW_AFTER_PEAK,
                                             min_valid_ratio=MIN_SEGMENT_VALID_RATIO)
        total_segments_before_filter += len(final_peak_indices) # Count peaks used for segmentation attempt

        for i, segment_data in enumerate(segments_7col):
            segment_id = f"{original_unique_id}_seg{i}"
            segment_filename = f"{segment_id}.npy"
            segment_filepath = os.path.join(segmented_data_dir, segment_filename)
            np.save(segment_filepath, segment_data)

            segment_info = {}
            segment_info['segment_id'] = segment_id
            segment_info['original_unique_id'] = original_unique_id
            segment_info['segment_filepath'] = segment_filepath
            
            # Find the original peak index that corresponds to this segment
            # This mapping is tricky if segments_7col is shorter than final_peak_indices due to MIN_SEGMENT_VALID_RATIO filter
            # For simplicity now, we might omit peak_index_in_original or need a more robust mapping
            # if i < len(final_peak_indices): # Basic check
            #    segment_info['peak_index_in_original'] = final_peak_indices[i]


            if not is_test_set:
                for col_label in ['gender', 'hold racket handed', 'play years', 'level', 'mode', 'player_id', 'cut_point']:
                    if col_label in row: segment_info[col_label] = row[col_label]
            all_segmented_info.append(segment_info)

    if all_segmented_info:
        segmented_df = pd.DataFrame(all_segmented_info)
        base_cols = ['segment_id', 'original_unique_id', 'segment_filepath']
        label_cols_ordered = ['gender', 'hold racket handed', 'play years', 'level', 'mode', 'player_id', 'cut_point']
        final_cols = base_cols
        if not is_test_set:
            existing_label_cols = [lc for lc in label_cols_ordered if lc in segmented_df.columns]
            final_cols.extend(existing_label_cols)
        else:
            existing_test_meta_cols = [lc for lc in ['mode', 'cut_point'] if lc in segmented_df.columns]
            final_cols.extend(existing_test_meta_cols)
        other_cols = [c for c in segmented_df.columns if c not in final_cols]
        final_cols.extend(other_cols)
        segmented_df = segmented_df[final_cols]
        segmented_df.to_csv(segmented_info_file_out, index=False)
        print(f"Saved {len(segmented_df)} valid segments info to {segmented_info_file_out}")
        print(f"Total peaks considered for segmentation (after dynamic height filter): {total_segments_before_filter}")
        print(f"Files with no initial candidate peaks: {files_with_no_peaks_at_all}")
        print(f"Files with no peaks after dynamic height filter: {files_with_no_peaks_after_height_filter}")

    else:
        print(f"No valid segments were generated for {info_file_original}.")

if __name__ == '__main__':
    print("Starting data preprocessing and segmentation...")
    # Process Training Data
    preprocess_and_segment_data(
        info_file_original=ORIGINAL_TRAIN_INFO_FILE,
        original_data_folder=ORIGINAL_TRAIN_DATA_FOLDER,
        segmented_data_dir=SEGMENTED_TRAIN_DATA_DIR,
        segmented_info_file_out=SEGMENTED_TRAIN_INFO_FILE,
        is_test_set=False
    )
    # Process Validation Data
    preprocess_and_segment_data(
        info_file_original=ORIGINAL_VALID_INFO_FILE,
        original_data_folder=ORIGINAL_TRAIN_DATA_FOLDER,
        segmented_data_dir=SEGMENTED_VALID_DATA_DIR,
        segmented_info_file_out=SEGMENTED_VALID_INFO_FILE,
        is_test_set=False
    )
    # Process Test Data
    preprocess_and_segment_data(
        info_file_original=ORIGINAL_TEST_INFO_FILE,
        original_data_folder=ORIGINAL_TEST_DATA_FOLDER,
        segmented_data_dir=SEGMENTED_TEST_DATA_DIR,
        segmented_info_file_out=SEGMENTED_TEST_INFO_FILE,
        is_test_set=True
    )
    print("Preprocessing finished.")
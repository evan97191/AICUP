# --- START OF FILE apply_smote_selectable_target.py ---
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from collections import Counter
from tqdm import tqdm
import joblib

def main(SMOTE_TARGET_COLUMN):
    # --- NEW: Configuration for SMOTE Target ---
    # Options: 'gender', 'hold racket handed', 'play years', 'level'
    # If 'play years' or 'level', ensure the original CSV has these columns with appropriate values.


    # --- Configuration ---
    SEGMENTED_DATA_BASE_DIR = "segmented_data_final"
    TRAIN_INFO_FILE_SEGMENTED = os.path.join(SEGMENTED_DATA_BASE_DIR, "train_info_segmented.csv")
    SEGMENTED_TRAIN_DATA_DIR = os.path.join(SEGMENTED_DATA_BASE_DIR, "train")

    # --- NEW: Configuration for SMOTE Target ---
    # Options: 'gender', 'hold racket handed', 'play years', 'level'
    # If 'play years' or 'level', ensure the original CSV has these columns with appropriate values.
    # SMOTE_TARGET_COLUMN = 'gender'  # <<<<----- 在這裡選擇要 SMOTE 的目標列名

    SMOTE_OUTPUT_DATA_BASE_DIR = f"smote_data_selectable_{SMOTE_TARGET_COLUMN}" # New output dir
    SMOTE_TRAIN_INFO_FILE = os.path.join(SMOTE_OUTPUT_DATA_BASE_DIR, "train_info_smote.csv")
    SMOTE_TRAIN_DATA_DIR = os.path.join(SMOTE_OUTPUT_DATA_BASE_DIR, "train_smote")

    # SMOTE Parameters
    SMOTE_RANDOM_STATE = 42
    SMOTE_SAMPLING_STRATEGY = 'auto' # 'auto', 'minority', or a dict
    SMOTE_K_NEIGHBORS = 5


    # -----------------------------------------

    # Mappings for multi-class targets if chosen
    # Ensure these columns exist in your segmented_info_file if you choose them as target
    LEVEL_MAP = {2: 0, 3: 1, 4: 2, 5: 3} # Maps original level values to 0-indexed classes
    PLAY_YEARS_MAP = {0: 0, 1: 1, 2: 2}   # Maps original play years values to 0-indexed classes
    # For binary targets like 'gender', we might map 1 (Male) -> 1, 2 (Female) -> 0
    GENDER_MAP = {1:1, 2:0} # Example if 'gender' column uses 1 and 2
    HANDEDNESS_MAP = {1:1, 2:0} # Example if 'hold racket handed' uses 1 and 2


    def apply_smote_to_segments(segmented_info_file, segmented_data_dir,
                                smote_output_info_file, smote_output_data_dir,
                                target_column_name, target_map=None):
        print(f"Loading segmented training data from {segmented_info_file}...")
        try:
            df_train_segmented = pd.read_csv(
                segmented_info_file,
                dtype={'original_unique_id': str, 'segment_id': str} # Ensure IDs are strings
            )
        except FileNotFoundError:
            print(f"Error: Segmented info file not found: {segmented_info_file}"); return
        if df_train_segmented.empty:
            print(f"Error: Segmented info file {segmented_info_file} is empty."); return

        os.makedirs(smote_output_data_dir, exist_ok=True)

        # Check if target column exists
        if target_column_name not in df_train_segmented.columns:
            print(f"Error: SMOTE target column '{target_column_name}' not found in {segmented_info_file}. Available columns: {df_train_segmented.columns.tolist()}")
            return

        print("Loading all segment .npy files for SMOTE...")
        all_segment_features = []
        # Store a temporary dataframe with only successfully loaded segments and their original indices
        successfully_loaded_segment_info = []

        for index, row in tqdm(df_train_segmented.iterrows(), total=len(df_train_segmented), desc="Loading .npy files"):
            try:
                segment_data = np.load(row['segment_filepath'])
                all_segment_features.append(segment_data.flatten())
                # Store the full row for later easy access to all original labels
                row_dict = row.to_dict()
                row_dict['_original_index_in_df'] = index # Keep track of original index for alignment
                successfully_loaded_segment_info.append(row_dict)
            except Exception as e:
                # print(f"Warning: Could not load or process {row['segment_filepath']}: {e}. Skipping.")
                continue
        
        if not all_segment_features:
            print("Error: No segment data loaded. SMOTE cannot be applied."); return

        X_original = np.array(all_segment_features)
        df_train_for_smote = pd.DataFrame(successfully_loaded_segment_info) # DF of successfully loaded segments
        
        print(f"Original feature matrix X shape: {X_original.shape}")
        print(f"DataFrame for SMOTE shape: {df_train_for_smote.shape}")

        # Prepare target variable (y) for SMOTE based on target_column_name
        y_target_for_smote = df_train_for_smote[target_column_name].copy()
        if target_map:
            y_target_for_smote = y_target_for_smote.map(target_map)

        # Handle potential NaNs in the target column after mapping
        if y_target_for_smote.isnull().any():
            print(f"Warning: Found NaN values in target column '{target_column_name}' after mapping. Dropping these rows.")
            nan_indices = y_target_for_smote[y_target_for_smote.isnull()].index
            # df_train_for_smote and X_original need to be filtered consistently
            df_train_for_smote.drop(nan_indices, inplace=True)
            X_original = np.delete(X_original, nan_indices, axis=0) # Delete rows from X
            y_target_for_smote.dropna(inplace=True)
            print(f"Shapes after dropping NaNs: X_original={X_original.shape}, df_train_for_smote={df_train_for_smote.shape}, y_target={y_target_for_smote.shape}")

        if X_original.shape[0] == 0 or len(y_target_for_smote) == 0:
            print("Error: No valid samples remaining after NaN filtering in target. SMOTE cannot proceed.")
            return

        y_original_target_values = y_target_for_smote.astype(int).values
        print(f"Class distribution for '{target_column_name}' before SMOTE: {Counter(y_original_target_values)}")

        if X_original.shape[0] != len(y_original_target_values):
            print(f"CRITICAL ERROR: Mismatch X samples ({X_original.shape[0]}) and y samples ({len(y_original_target_values)}).")
            return

        # Adjust k_neighbors if any class has too few samples
        class_counts = Counter(y_original_target_values)
        min_class_count = min(class_counts.values()) if class_counts else 0
        
        # k_neighbors must be less than the number of samples in the smallest class.
        # If min_class_count is 1, SMOTE cannot be applied with k_neighbors >= 1.
        # If min_class_count <= k_neighbors, reduce k_neighbors.
        smote_k_actual = SMOTE_K_NEIGHBORS
        if min_class_count <= SMOTE_K_NEIGHBORS :
            if min_class_count <=1 :
                print(f"Warning: Minority class for '{target_column_name}' has {min_class_count} sample(s). SMOTE cannot be applied with k_neighbors > 0.")
                print("Skipping SMOTE for this target. Using original data.")
                X_resampled, y_resampled_target = X_original.copy(), y_original_target_values.copy() # Use original data
            else:
                smote_k_actual = min_class_count - 1
                print(f"Warning: Minority class count ({min_class_count}) is <= k_neighbors ({SMOTE_K_NEIGHBORS}). Adjusting k_neighbors to {smote_k_actual}.")
        
        if min_class_count > 1 : # Only apply SMOTE if all classes have at least 2 samples
            print(f"Applying SMOTE on '{target_column_name}' with strategy: {SMOTE_SAMPLING_STRATEGY}, k_neighbors: {smote_k_actual}...")
            smote = SMOTE(random_state=SMOTE_RANDOM_STATE,
                        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                        k_neighbors=smote_k_actual)
            try:
                X_resampled, y_resampled_target = smote.fit_resample(X_original, y_original_target_values)
            except ValueError as e:
                print(f"Error during SMOTE: {e}. Using original data instead.")
                X_resampled, y_resampled_target = X_original.copy(), y_original_target_values.copy()
        else: # If min_class_count <=1, SMOTE was skipped
            pass # X_resampled and y_resampled_target are already copies of original

        print(f"Feature matrix X shape after SMOTE: {X_resampled.shape}")
        print(f"Class distribution for '{target_column_name}' after SMOTE: {Counter(y_resampled_target)}")

        # Reconstruct DataFrame
        num_original_samples = X_original.shape[0]
        df_smote_info_list = []
        
        # Create inverse maps if needed (only for the target column if it was mapped)
        inv_target_map = None
        if target_map:
            inv_target_map = {v: k for k, v in target_map.items()}

        # SEQUENCE_LENGTH and INPUT_CHANNELS for reshaping
        # It's better to get these from a reliable source (e.g., first loaded .npy file's shape)
        # Assuming they are consistent.
        try:
            sample_seg_shape = np.load(df_train_for_smote.iloc[0]['segment_filepath']).shape
            cfg_sequence_length = sample_seg_shape[0]
            cfg_input_channels = sample_seg_shape[1]
            print(f"Determined segment shape for reshaping: ({cfg_sequence_length}, {cfg_input_channels})")
        except Exception:
            print("Could not determine segment shape. Using hardcoded fallback (120, 7). This might be wrong.")
            cfg_sequence_length = 120 # Fallback, ensure this is correct
            cfg_input_channels = 7   # Fallback

        print(f"Reconstructing segments and saving .npy files for {X_resampled.shape[0]} samples...")
        for i in tqdm(range(X_resampled.shape[0]), desc="Saving SMOTE'd data"):
            segment_data_flat = X_resampled[i]
            segment_data_reshaped = segment_data_flat.reshape(cfg_sequence_length, cfg_input_channels)

            new_segment_id = f"smote_seg_{i}"
            new_segment_filename = f"{new_segment_id}.npy"
            new_segment_filepath = os.path.join(smote_output_data_dir, new_segment_filename)
            np.save(new_segment_filepath, segment_data_reshaped)

            current_info = {}
            current_info['segment_id'] = new_segment_id
            current_info['segment_filepath'] = new_segment_filepath
            
            # Store the SMOTE'd target value (mapped if a map was used)
            current_info[f'{target_column_name}_smote_mapped'] = y_resampled_target[i]
            # Store the original scale value of the SMOTE'd target
            if inv_target_map:
                current_info[target_column_name] = inv_target_map.get(y_resampled_target[i], np.nan)
            else: # If no map was used (e.g. target was already 0/1), it's the same
                current_info[target_column_name] = y_resampled_target[i]


            if i < num_original_samples:
                # This is an original sample, copy all its original info
                # Use the original index stored in df_train_for_smote to get the correct row
                original_df_row_index = df_train_for_smote.iloc[i]['_original_index_in_df']
                original_row_data = df_train_segmented.loc[original_df_row_index]

                current_info['original_unique_id'] = str(original_row_data['original_unique_id'])
                for col in df_train_segmented.columns:
                    if col not in [target_column_name, 'segment_id', 'segment_filepath', '_original_index_in_df', f'{target_column_name}_smote_mapped']:
                        current_info[col] = original_row_data[col]
            else:
                # This is a synthetic sample
                # Assign the SMOTE'd target. For other labels, copy from a reference original sample.
                target_val_for_synthetic = y_resampled_target[i]
                
                # Find original samples that had this target_val_for_synthetic (before SMOTE)
                # Need to use df_train_for_smote and its *original* target column values before mapping (if mapped)
                # or its mapped values if no map was used initially for y_original_target_values
                
                # df_train_for_smote already has y_target_for_smote (which is mapped)
                # So we find rows in df_train_for_smote where its mapped target matches target_val_for_synthetic
                ref_column_for_matching = y_target_for_smote.name # This is the mapped target series
                
                matching_original_samples_df = df_train_for_smote[y_target_for_smote == target_val_for_synthetic]

                if not matching_original_samples_df.empty:
                    ref_original_row_data = matching_original_samples_df.iloc[0] # Take the first match
                    current_info['original_unique_id'] = str(ref_original_row_data['original_unique_id']) + "_synthetic"
                    for col in df_train_segmented.columns:
                        if col not in [target_column_name, 'segment_id', 'segment_filepath', '_original_index_in_df', f'{target_column_name}_smote_mapped']:
                            current_info[col] = ref_original_row_data[col]
                else:
                    # This case should be rare if SMOTE works on existing classes
                    print(f"Warning: No reference original sample for synthetic sample {i} with target {target_val_for_synthetic}")
                    current_info['original_unique_id'] = "unknown_synthetic"
                    # Fill other columns with NaN or default
                    for col in df_train_segmented.columns:
                        if col not in [target_column_name, 'segment_id', 'segment_filepath', '_original_index_in_df', f'{target_column_name}_smote_mapped']:
                            current_info[col] = np.nan
            df_smote_info_list.append(current_info)

        df_smote_final = pd.DataFrame(df_smote_info_list)
        df_smote_final = shuffle(df_smote_final, random_state=SMOTE_RANDOM_STATE)
        
        # Ensure all original columns are present, fill NaNs if any were introduced for synthetic samples' non-target labels
        for original_col in df_train_segmented.columns:
            if original_col not in df_smote_final.columns and original_col not in ['_original_index_in_df']:
                df_smote_final[original_col] = np.nan # Add column if missing
                print(f"Added missing column {original_col} to SMOTE'd df, will be NaN for many rows.")
        
        # Reorder columns to be similar to original, with new SMOTE cols at the end
        cols_order = [c for c in df_train_segmented.columns if c not in ['_original_index_in_df']]
        if f'{target_column_name}_smote_mapped' not in cols_order:
            cols_order.append(f'{target_column_name}_smote_mapped')
        
        # Ensure all columns in df_smote_final are in cols_order, add any new ones
        for c in df_smote_final.columns:
            if c not in cols_order:
                cols_order.append(c)
        
        # Filter cols_order to only existing columns in df_smote_final to prevent KeyError
        cols_order = [c for c in cols_order if c in df_smote_final.columns]

        df_smote_final = df_smote_final[cols_order]

        df_smote_final.to_csv(smote_output_info_file, index=False)
        print(f"Saved {len(df_smote_final)} SMOTE'd samples info to {smote_output_info_file}")


   
    # Determine which map to use based on SMOTE_TARGET_COLUMN
    active_target_map = None
    if SMOTE_TARGET_COLUMN == 'level':
        active_target_map = LEVEL_MAP
    elif SMOTE_TARGET_COLUMN == 'play years':
        active_target_map = PLAY_YEARS_MAP
    elif SMOTE_TARGET_COLUMN == 'gender': # Assuming gender is 1,2 needs mapping to 0,1
        active_target_map = GENDER_MAP
    elif SMOTE_TARGET_COLUMN == 'hold racket handed': # Assuming handedness is 1,2 needs mapping to 0,1
        active_target_map = HANDEDNESS_MAP
    # Add more mappings if other columns are targeted and need it

    apply_smote_to_segments(
        segmented_info_file=TRAIN_INFO_FILE_SEGMENTED,
        segmented_data_dir=SEGMENTED_TRAIN_DATA_DIR, # Used for potential path joining if needed
        smote_output_info_file=SMOTE_TRAIN_INFO_FILE,
        smote_output_data_dir=SMOTE_TRAIN_DATA_DIR,
        target_column_name=SMOTE_TARGET_COLUMN,
        target_map=active_target_map
    )


if __name__ == '__main__':
    main('gender')
    main('hold racket handed')
    main('play years') 
    main('level')

    print("SMOTE application finished.")
    # --- END OF FILE apply_smote_selectable_target.py ---
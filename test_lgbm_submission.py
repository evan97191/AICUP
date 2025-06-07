# --- START OF FILE test_lgbm_multi_cnn_submission.py ---
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import lightgbm as lgb
import joblib
from tqdm import tqdm
import multiprocessing

from model import BaseCNN1D # For feature extraction
from utils import pad_or_truncate, adjust_sum_to_one

# --- Configuration ---
# Paths for Multi-CNN feature extraction part
BASE_PRETRAINED_CNN_DIR_TEST = "trained_separate_models" # Dir where task-specific CNNs are
CNN_MODEL_PATHS_TEST = { # Must match extract_cnn_features_multi_model.py
    'gender': os.path.join(BASE_PRETRAINED_CNN_DIR_TEST, "gender", "cnn1d_gender_best.pth"),
    'handedness': os.path.join(BASE_PRETRAINED_CNN_DIR_TEST, "handedness", "cnn1d_handedness_best.pth"),
    'play_years': os.path.join(BASE_PRETRAINED_CNN_DIR_TEST, "play_years", "cnn1d_play_years_best.pth"),
    'level': os.path.join(BASE_PRETRAINED_CNN_DIR_TEST, "level", "cnn1d_level_best.pth"),
}
NUM_CLASSES_OF_LOADED_CNNS_TEST = { # Must match extract_cnn_features_multi_model.py
    'gender': 1, 'handedness': 1, 'play_years': 3, 'level': 4,
}
SCALER_PATH_CNN_GLOBAL = os.path.join("scalers_for_separate_models", "global_scaler_all_features.joblib")

# Paths for LightGBM models (trained on multi-CNN features)
LGBM_MODEL_DIR_MULTI_CNN = "lgbm_models_from_multi_cnn"

# Test data info
SEGMENTED_DATA_BASE_DIR_TEST = "segmented_data_final"
TEST_INFO_FILE_SEGMENTED = os.path.join(SEGMENTED_DATA_BASE_DIR_TEST, "test_info_segmented.csv")
ORIGINAL_TEST_INFO_FILE = os.path.join("39_Test_Dataset", "test_info.csv")

OUTPUT_SUBMISSION_FILE = "submission_lgbm_from_multi_cnn.csv" # New output

# CNN Feature Extraction Config
WINDOW_BEFORE_PEAK = 29; WINDOW_AFTER_PEAK = 30
SEQUENCE_LENGTH = WINDOW_BEFORE_PEAK + 1 + WINDOW_AFTER_PEAK
INPUT_CHANNELS_CNN_EXT = 7 # Input to each CNN
BATCH_SIZE_CNN_FEAT_EXT = 256

# LGBM Task Config (must match training)
TASKS_LGBM_SUBMISSION = {
    'gender': {'model_file': f"lgbm_multi_cnn_feat_gender.txt", 'type': 'binary', 'output_indices': [0]},
    'handedness': {'model_file': f"lgbm_multi_cnn_feat_handedness.txt", 'type': 'binary', 'output_indices': [1]},
    'play_years': {'model_file': f"lgbm_multi_cnn_feat_play_years.txt", 'type': 'multiclass', 'num_classes': 3, 'output_indices': [2,3,4]},
    'level': {'model_file': f"lgbm_multi_cnn_feat_level.txt", 'type': 'multiclass', 'num_classes': 4, 'output_indices': [5,6,7,8]}
}
TOTAL_OUTPUT_COLUMNS_SUBMISSION = 9
DECIMAL_PLACES_BINARY = 2; DECIMAL_PLACES_MULTI = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset for CNN Feature Extraction (same as in extract_cnn_features_multi_model.py) ---
class TestFeatureExtractionDataset(Dataset):
    def __init__(self, info_df_segmented, sequence_length, num_input_channels, scaler):
        self.info_df = info_df_segmented.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.num_input_channels = num_input_channels
        self.scaler = scaler
    def __len__(self): return len(self.info_df)
    def __getitem__(self, idx):
        row = self.info_df.iloc[idx]
        segment_filepath = row['segment_filepath']
        original_unique_id = str(row.get('original_unique_id', 'N/A'))
        segment_id = str(row.get('segment_id', 'N/A'))
        try: segment_data = np.load(segment_filepath)
        except Exception: segment_data = np.zeros((self.sequence_length, self.num_input_channels), dtype=np.float32)
        if segment_data.shape[0] != self.sequence_length or segment_data.shape[1] != self.num_input_channels:
            segment_data = pad_or_truncate(segment_data, self.sequence_length, self.num_input_channels)
        if self.scaler: segment_data = self.scaler.transform(segment_data)
        segment_data_transposed = segment_data.T
        sequence_tensor = torch.from_numpy(segment_data_transposed).float()
        return sequence_tensor, original_unique_id, segment_id

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print(f"Using device: {DEVICE} for CNN feature extraction.")

    # 1. Load Scaler and ALL Pre-trained CNNs for feature extraction
    try:
        scaler_cnn_global = joblib.load(SCALER_PATH_CNN_GLOBAL)
        cnn_extractors = {}
        for task_name, model_path in CNN_MODEL_PATHS_TEST.items():
            num_classes = NUM_CLASSES_OF_LOADED_CNNS_TEST[task_name]
            cnn_model = BaseCNN1D(input_channels=INPUT_CHANNELS_CNN_EXT, num_classes_for_task=num_classes).to(DEVICE)
            cnn_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            cnn_model.eval()
            cnn_extractors[task_name] = cnn_model
        print("All CNN feature extractors and global scaler loaded.")
    except Exception as e: print(f"Error loading CNNs/scaler: {e}"); exit()

    # 2. Load segmented test data info
    try: df_test_info_segmented = pd.read_csv(TEST_INFO_FILE_SEGMENTED, dtype={'original_unique_id': str, 'segment_id': str})
    except FileNotFoundError: print(f"Error: {TEST_INFO_FILE_SEGMENTED} not found."); exit()
    if df_test_info_segmented.empty: print(f"{TEST_INFO_FILE_SEGMENTED} is empty."); exit()

    test_cnn_dataset_submission = TestFeatureExtractionDataset(df_test_info_segmented, SEQUENCE_LENGTH, INPUT_CHANNELS_CNN_EXT, scaler_cnn_global)
    test_cnn_loader_submission = DataLoader(test_cnn_dataset_submission, batch_size=BATCH_SIZE_CNN_FEAT_EXT, shuffle=False, num_workers=0)

    # 3. Extract Concatenated CNN features for the test set
    print("Extracting concatenated CNN features from test segments...")
    all_concatenated_cnn_features_list = []
    all_original_ids_for_lgbm_list = []
    # all_segment_ids_for_lgbm_list = [] # Not strictly needed for prediction if df is made from features directly
    with torch.no_grad():
        for sequences, original_ids_batch, segment_ids_batch in tqdm(test_cnn_loader_submission, desc="Extracting Multi-CNN Features"):
            sequences = sequences.to(DEVICE)
            batch_features_from_each_cnn = []
            # Ensure cnn_extractors keys are iterated in a consistent order if feature order matters
            # (it does for LGBM if trained on features in a specific order)
            # Using CNN_MODEL_PATHS_TEST.keys() ensures this consistent order.
            for task_name_src in CNN_MODEL_PATHS_TEST.keys():
                cnn_model = cnn_extractors[task_name_src]
                features_task = cnn_model(sequences, extract_features=True)
                batch_features_from_each_cnn.append(features_task.cpu().numpy())
            
            concatenated_batch_features = np.concatenate(batch_features_from_each_cnn, axis=1)
            all_concatenated_cnn_features_list.append(concatenated_batch_features)
            all_original_ids_for_lgbm_list.extend(list(original_ids_batch))
            # all_segment_ids_for_lgbm_list.extend(list(segment_ids_batch))

    if not all_concatenated_cnn_features_list: print("Error: No CNN features extracted."); exit()
    test_concatenated_cnn_features_np = np.concatenate(all_concatenated_cnn_features_list, axis=0)
    
    # Create a DataFrame for these features. The column order must match how LGBM was trained.
    lgbm_feature_column_names = []
    for task_name_src_lgbm in CNN_MODEL_PATHS_TEST.keys(): # Consistent order
        num_feats_per_cnn_lgbm = cnn_extractors[task_name_src_lgbm].feature_extractor_fc.out_features
        for i in range(num_feats_per_cnn_lgbm):
            lgbm_feature_column_names.append(f'{task_name_src_lgbm}_cnn_feat_{i}')
            
    df_test_lgbm_features = pd.DataFrame(test_concatenated_cnn_features_np, columns=lgbm_feature_column_names)
    # Add original_unique_id to df_test_lgbm_features for grouping
    df_test_lgbm_features['original_unique_id'] = all_original_ids_for_lgbm_list
    # df_test_lgbm_features['segment_id'] = all_segment_ids_for_lgbm_list # If needed
    print(f"Concatenated CNN features extracted for {len(df_test_lgbm_features)} test segments.")


    # 4. Load trained LightGBM models
    lgbm_prediction_models = {}
    for task_name_lgbm, config_lgbm in TASKS_LGBM_SUBMISSION.items():
        model_path_lgbm = os.path.join(LGBM_MODEL_DIR_MULTI_CNN, config_lgbm['model_file'])
        try:
            bst_lgbm = lgb.Booster(model_file=model_path_lgbm)
            lgbm_prediction_models[task_name_lgbm] = bst_lgbm
            print(f"LGBM model for task {task_name_lgbm} loaded.")
        except Exception as e: print(f"Error loading LGBM model {task_name_lgbm}: {e}"); exit()

    # 5. Predict with LightGBM models per segment using concatenated features
    lgbm_probas_per_original_id = {}
    # Features for LGBM are all columns except 'original_unique_id', 'segment_id'
    X_test_for_lgbm_models = df_test_lgbm_features[lgbm_feature_column_names].values

    print("Predicting with LightGBM models using concatenated features...")
    for task_name_lgbm_pred, lgbm_booster_pred in tqdm(lgbm_prediction_models.items(), desc="LGBM Task Predictions (Multi-CNN Feat)"):
        task_config_pred = TASKS_LGBM_SUBMISSION[task_name_lgbm_pred]
        
        task_probas_segments_lgbm = lgbm_booster_pred.predict(X_test_for_lgbm_models)
        
        if task_config_pred['type'] == 'binary':
            if task_probas_segments_lgbm.ndim > 1 and task_probas_segments_lgbm.shape[1] == 2:
                task_probas_segments_lgbm = task_probas_segments_lgbm[:, 1]
            elif task_probas_segments_lgbm.ndim > 1 and task_probas_segments_lgbm.shape[1] == 1:
                task_probas_segments_lgbm = task_probas_segments_lgbm.squeeze(axis=1)
        # For multiclass, predict() directly gives (N, num_classes) probabilities

        for i in range(len(df_test_lgbm_features)):
            original_id = df_test_lgbm_features.iloc[i]['original_unique_id']
            segment_prediction_for_task_lgbm = task_probas_segments_lgbm[i]
            if original_id not in lgbm_probas_per_original_id:
                lgbm_probas_per_original_id[original_id] = {t: [] for t in TASKS_LGBM_SUBMISSION.keys()}
            lgbm_probas_per_original_id[original_id][task_name_lgbm_pred].append(segment_prediction_for_task_lgbm)

    # 6. Aggregate, Format, and Save Submission (same logic as your test_final_strict_sum.py)
    print("Aggregating final predictions and formatting for submission...")
    final_submission_rows_multi_cnn_lgbm = []
    try:
        original_test_info_df_final = pd.read_csv(ORIGINAL_TEST_INFO_FILE, dtype={'unique_id': str})
        original_test_info_df_final.dropna(subset=['unique_id'], inplace=True)
        original_test_info_df_final['unique_id'] = original_test_info_df_final['unique_id'].apply(lambda x: str(int(float(x))) if pd.notna(x) and '.' in str(x) and str(x).replace('.', '', 1).isdigit() else str(x))
    except FileNotFoundError: print(f"Error: {ORIGINAL_TEST_INFO_FILE} not found."); exit()

    for original_id_to_submit_final in original_test_info_df_final['unique_id']:
        final_output_row_submission = np.zeros(TOTAL_OUTPUT_COLUMNS_SUBMISSION, dtype=np.float32)
        if original_id_to_submit_final in lgbm_probas_per_original_id:
            task_probas_for_id = lgbm_probas_per_original_id[original_id_to_submit_final]
            for task_name_sub, config_sub in TASKS_LGBM_SUBMISSION.items():
                output_indices_sub = config_sub['output_indices']
                if task_probas_for_id[task_name_sub]:
                    aggregated_task_probas_sub = np.mean(np.array(task_probas_for_id[task_name_sub]), axis=0)
                    if config_sub['type'] == 'multiclass':
                        adjusted_probs_sub = adjust_sum_to_one(aggregated_task_probas_sub, DECIMAL_PLACES_MULTI)
                        final_output_row_submission[output_indices_sub[0] : output_indices_sub[0] + len(adjusted_probs_sub)] = adjusted_probs_sub
                    else: # Binary
                        rounded_prob_sub = np.round(aggregated_task_probas_sub.item() if aggregated_task_probas_sub.ndim > 0 else aggregated_task_probas_sub, DECIMAL_PLACES_BINARY)
                        final_output_row_submission[output_indices_sub[0]] = rounded_prob_sub
                else: # Fallback for missing task prediction for an ID
                    if config_sub['type'] == 'multiclass': final_output_row_submission[output_indices_sub[0]:output_indices_sub[0]+config_sub['num_classes']] = [1.0/config_sub['num_classes']] * config_sub['num_classes']
                    else: final_output_row_submission[output_indices_sub[0]] = 0.5
        else: # Fallback for missing original_id entirely
            # ... (your default filling logic from previous script) ...
            final_output_row_submission[TASKS_LGBM_SUBMISSION['gender']['output_indices'][0]] = 0.50
            final_output_row_submission[TASKS_LGBM_SUBMISSION['handedness']['output_indices'][0]] = 0.50
            py_indices_def = TASKS_LGBM_SUBMISSION['play_years']['output_indices']
            py_num_cls_def = TASKS_LGBM_SUBMISSION['play_years']['num_classes']
            final_output_row_submission[py_indices_def[0]:py_indices_def[-1]+1] = [np.round(1/py_num_cls_def, DECIMAL_PLACES_MULTI)] * py_num_cls_def
            final_output_row_submission[py_indices_def[-1]] = np.round(1.0 - np.sum(final_output_row_submission[py_indices_def[0]:py_indices_def[-1]]), DECIMAL_PLACES_MULTI)
            lvl_indices_def = TASKS_LGBM_SUBMISSION['level']['output_indices']
            lvl_num_cls_def = TASKS_LGBM_SUBMISSION['level']['num_classes']
            final_output_row_submission[lvl_indices_def[0]:lvl_indices_def[-1]+1] = [np.round(1/lvl_num_cls_def, DECIMAL_PLACES_MULTI)] * lvl_num_cls_def
            final_output_row_submission[lvl_indices_def[-1]] = np.round(1.0 - np.sum(final_output_row_submission[lvl_indices_def[0]:lvl_indices_def[-1]]), DECIMAL_PLACES_MULTI)

        final_submission_rows_multi_cnn_lgbm.append([original_id_to_submit_final] + final_output_row_submission.tolist())

    submission_df_final_lgbm = pd.DataFrame(final_submission_rows_multi_cnn_lgbm, columns=['unique_id', 'gender', 'hold racket handed',
                                                                  'play years_0', 'play years_1', 'play years_2',
                                                                  'level_2', 'level_3', 'level_4', 'level_5'])
    submission_df_final_lgbm.to_csv(OUTPUT_SUBMISSION_FILE, index=False)
    print(f"LGBM (Multi-CNN Feat) Submission file saved to {OUTPUT_SUBMISSION_FILE}")
# --- END OF FILE test_lgbm_multi_cnn_submission.py ---
# --- START OF FILE extract_cnn_features_multi_model.py ---
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import joblib
from tqdm import tqdm

from model import BaseCNN1D # Your CNN model definition
# from train_multitask_separate_models import TaskSpecificSegmentedDataset # Or a simplified version
from utils import pad_or_truncate

# --- Configuration ---
# Paths to segmented data info CSVs
TRAIN_INFO_FILE_SEGMENTED = os.path.join("segmented_data_final", "train_info_segmented.csv")
VALID_INFO_FILE_SEGMENTED = os.path.join("segmented_data_final", "valid_info_segmented.csv")
TEST_INFO_FILE_SEGMENTED = os.path.join("segmented_data_final", "test_info_segmented.csv")

# Paths to the pre-trained CNN model weights FOR EACH TASK
BASE_PRETRAINED_CNN_DIR = "trained_separate_models"
CNN_MODEL_PATHS = {
    'gender': os.path.join(BASE_PRETRAINED_CNN_DIR, "gender", "cnn1d_gender_best.pth"),
    'handedness': os.path.join(BASE_PRETRAINED_CNN_DIR, "handedness", "cnn1d_handedness_best.pth"),
    'play_years': os.path.join(BASE_PRETRAINED_CNN_DIR, "play_years", "cnn1d_play_years_best.pth"),
    'level': os.path.join(BASE_PRETRAINED_CNN_DIR, "level", "cnn1d_level_best.pth"),
}
# Output classes of each pre-trained CNN (needed to load them correctly)
# This should match how they were saved by train_multitask_separate_models.py
NUM_CLASSES_OF_LOADED_CNNS = {
    'gender': 1,
    'handedness': 1,
    'play_years': 3,
    'level': 4,
}

SCALER_PATH = os.path.join("scalers_for_separate_models", "global_scaler_all_features.joblib")

# Output directory for concatenated extracted features
FEATURE_OUTPUT_DIR = "lgbm_features_from_multi_cnn" # New output dir
os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

# Parameters matching CNN training
WINDOW_BEFORE_PEAK = 29
WINDOW_AFTER_PEAK = 30
SEQUENCE_LENGTH = WINDOW_BEFORE_PEAK + 1 + WINDOW_AFTER_PEAK
INPUT_CHANNELS_CNN = 7 # Input channels for each CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_FEATURES = 256

# --- Dataset for Feature Extraction (Simplified - same as in test_lgbm_submission.py) ---
class FeatureExtractionDataset(Dataset):
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


def extract_features_from_multiple_cnns(dataloader, cnn_models_dict, device):
    for model in cnn_models_dict.values():
        model.eval() # IMPORTANT: Set all models to evaluation mode

    # To store features from each model before concatenation
    # {segment_id: {task_name: feature_vector}}
    # No, simpler: {segment_id: [feature_cnn1, feature_cnn2, ...]} then concatenate later
    # Or even simpler: extract for all models, then hstack features based on common segment_id
    
    # Let's do it per batch for memory efficiency:
    # For each batch, get features from all CNNs, then concatenate for that batch.
    all_concatenated_features = []
    all_original_ids = []
    all_segment_ids = []

    with torch.no_grad():
        for sequences, original_ids_batch, segment_ids_batch in tqdm(dataloader, desc="Extracting Multi-CNN Features"):
            sequences = sequences.to(device)
            
            batch_features_from_each_cnn = []
            for task_name, cnn_model in cnn_models_dict.items():
                # Assuming model's forward method has 'extract_features=True'
                features_task = cnn_model(sequences, extract_features=True) # (batch_size, 64)
                batch_features_from_each_cnn.append(features_task.cpu().numpy())
            
            # Concatenate features from all CNNs for this batch horizontally
            # If each CNN gives (batch, 64), and we have 4 CNNs, result is (batch, 256)
            concatenated_batch_features = np.concatenate(batch_features_from_each_cnn, axis=1)
            
            all_concatenated_features.append(concatenated_batch_features)
            all_original_ids.extend(list(original_ids_batch))
            all_segment_ids.extend(list(segment_ids_batch))
            
    return np.concatenate(all_concatenated_features, axis=0), all_original_ids, all_segment_ids

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # Load Scaler
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded from {SCALER_PATH}")
    except Exception as e:
        print(f"Error loading scaler from {SCALER_PATH}: {e}"); exit()

    # Load all pre-trained CNN models
    cnn_models_for_extraction = {}
    print("Loading pre-trained CNN models for feature extraction...")
    for task_name, model_path in CNN_MODEL_PATHS.items():
        num_classes = NUM_CLASSES_OF_LOADED_CNNS[task_name]
        cnn_model = BaseCNN1D(input_channels=INPUT_CHANNELS_CNN, num_classes_for_task=num_classes).to(DEVICE)
        try:
            cnn_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            cnn_models_for_extraction[task_name] = cnn_model
            print(f"Loaded CNN for task '{task_name}' from {model_path}")
        except Exception as e:
            print(f"Error loading pre-trained CNN model for task '{task_name}': {e}"); exit()


    datasets_to_process = {
        'train': TRAIN_INFO_FILE_SEGMENTED,
        'valid': VALID_INFO_FILE_SEGMENTED,
        'test': TEST_INFO_FILE_SEGMENTED
    }

    for dset_name, info_file in datasets_to_process.items():
        print(f"\nProcessing dataset: {dset_name} from {info_file}")
        if not os.path.exists(info_file):
            print(f"Info file {info_file} not found. Skipping {dset_name}.")
            continue
        
        df_info = pd.read_csv(info_file, dtype={'original_unique_id': str, 'segment_id': str})
        if df_info.empty:
            print(f"Info file {info_file} is empty. Skipping {dset_name}.")
            continue

        feature_dataset = FeatureExtractionDataset(df_info, SEQUENCE_LENGTH, INPUT_CHANNELS_CNN, scaler)
        # num_workers=0 for feature extraction might be simpler to debug, but can be slow.
        # If using >0, ensure all components are pickleable.
        feature_dataloader = DataLoader(feature_dataset, batch_size=BATCH_SIZE_FEATURES, shuffle=False, num_workers=0)

        concatenated_features_np, original_ids_np, segment_ids_np = extract_features_from_multiple_cnns(
            feature_dataloader, cnn_models_for_extraction, DEVICE
        )
        
        print(f"Extracted concatenated features shape for {dset_name}: {concatenated_features_np.shape}")

        # Save features along with identifiers
        # Create column names that indicate the source CNN task for the features
        feature_column_names = []
        for task_name_src in cnn_models_for_extraction.keys(): # Iterate in defined order
            # Assuming each CNN extracts 64 features (from BaseCNN1D's feature_extractor_fc)
            num_feats_per_cnn = cnn_models_for_extraction[task_name_src].feature_extractor_fc.out_features
            for i in range(num_feats_per_cnn):
                feature_column_names.append(f'{task_name_src}_cnn_feat_{i}')

        features_df = pd.DataFrame(concatenated_features_np, columns=feature_column_names)
        features_df['original_unique_id'] = original_ids_np
        features_df['segment_id'] = segment_ids_np
        
        id_cols = ['original_unique_id', 'segment_id']
        # feature_cols already defined by feature_column_names
        features_df = features_df[id_cols + feature_column_names]


        output_path = os.path.join(FEATURE_OUTPUT_DIR, f"{dset_name}_multi_cnn_features.csv")
        features_df.to_csv(output_path, index=False)
        print(f"Concatenated CNN features for {dset_name} saved to {output_path}")

    print("\nMulti-CNN feature extraction complete.")
# --- END OF FILE extract_cnn_features_multi_model.py ---
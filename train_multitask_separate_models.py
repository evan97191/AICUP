# --- START OF FILE train_multitask_separate_models.py ---
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import joblib
from tqdm import tqdm
import multiprocessing
import random
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from model import BaseCNN1D # Use the parameterized BaseCNN1D
from utils import pad_or_truncate

# --- Configuration ---
SEGMENTED_DATA_BASE_DIR = "segmented_data_final" # From preprocess_data.py
TRAIN_INFO_FILE_SEGMENTED = os.path.join(SEGMENTED_DATA_BASE_DIR, "train_info_segmented.csv")
VALID_INFO_FILE_SEGMENTED = os.path.join(SEGMENTED_DATA_BASE_DIR, "valid_info_segmented.csv")

#'gender' 'play years' 'level'
# CLASS = "level"
# SEGMENTED_DATA_BASE_DIR = "segmented_data_final" # From preprocess_data.py
# TRAIN_INFO_FILE_SEGMENTED = os.path.join(f"smote_data_selectable_{CLASS}", "train_info_smote.csv")
# VALID_INFO_FILE_SEGMENTED = os.path.join(SEGMENTED_DATA_BASE_DIR, "valid_info_segmented.csv")

BASE_MODEL_SAVE_DIR = "trained_separate_models" # Directory to save all models
BASE_SCALER_SAVE_DIR = "scalers_for_separate_models" # Dir for scalers
CONFUSION_MATRIX_BASE_DIR = "confusion_matrices_separate"

WINDOW_BEFORE_PEAK = 29
WINDOW_AFTER_PEAK = 30
SEQUENCE_LENGTH = WINDOW_BEFORE_PEAK + 1 + WINDOW_AFTER_PEAK
INPUT_CHANNELS = 7 # Ax,Ay,Az,Gx,Gy,Gz, DetrendedAccSqSum

# Task-specific configurations will be defined within the training loop for each task
BATCH_SIZE = 1024 # Can be adjusted per task if needed
BASE_LEARNING_RATE = 0.0005
BASE_NUM_EPOCHS = 40 # Can be adjusted per task
BASE_WEIGHT_DECAY = 1e-5

# Augmentation (applied to the training data for all tasks)
AUGMENT_TRAIN_DATA = True
OCCLUSION_PROBABILITY = 0.5
MAX_OCCLUSION_LENGTH_RATIO = 0.31415
OCCLUSION_VALUE = 0.0

# --- Dataset Class for Segmented Data (Task-Specific Label Preparation) ---
class TaskSpecificSegmentedDataset(Dataset):
    def __init__(self, info_df_segmented, sequence_length, num_input_channels, scaler,
                 task_name, is_training_set=False, augment_data=False, **aug_params):
        self.info_df = info_df_segmented.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.num_input_channels = num_input_channels
        self.scaler = scaler # Scaler is fit once on all input features
        self.task_name = task_name
        self.is_training_set = is_training_set
        self.augment_data = augment_data
        self.aug_params = aug_params

        self._prepare_task_labels()

    def _prepare_task_labels(self):
        self.task_labels = []
        if self.task_name == 'gender':
            # gender: 1:Male, 2:Female -> map to 0 (Female), 1 (Male)
            self.task_labels = (self.info_df['gender'].apply(lambda x: 1.0 if x == 1 else 0.0)).astype(np.float32).values
        elif self.task_name == 'handedness':
            # hold racket handed: 1:Right, 2:Left -> map to 0 (Left), 1 (Right)
            self.task_labels = (self.info_df['hold racket handed'].apply(lambda x: 1.0 if x == 1 else 0.0)).astype(np.float32).values
        elif self.task_name == 'play_years':
            # play years: 0:低, 1:中, 2:高 (already 0,1,2) -> target is class index
            self.task_labels = self.info_df['play years'].astype(np.int64).values # CrossEntropyLoss needs int64
        elif self.task_name == 'level':
            # level: 2:甲, 3:乙, 4:青國, 5:青選 -> map to 0,1,2,3
            level_map = {2: 0, 3: 1, 4: 2, 5: 3}
            self.task_labels = self.info_df['level'].map(level_map).astype(np.int64).values
        else:
            raise ValueError(f"Unknown task_name: {self.task_name}")

    def _apply_temporal_occlusion(self, sequence_data):
        if random.random() < self.aug_params.get('occlusion_prob', 0.0):
            max_len = int(self.sequence_length * self.aug_params.get('max_occlusion_len_ratio', 0.0))
            if max_len < 1: return sequence_data
            occlusion_length = random.randint(1, max_len)
            start_point = random.randint(0, self.sequence_length - occlusion_length)
            occlusion_val = self.aug_params.get('occlusion_val', 0.0)
            sequence_data_occluded = sequence_data.copy()
            sequence_data_occluded[start_point : start_point + occlusion_length, :] = occlusion_val
            return sequence_data_occluded
        return sequence_data

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        row = self.info_df.iloc[idx]
        segment_filepath = row['segment_filepath']
        try:
            segment_data = np.load(segment_filepath)
        except Exception:
            segment_data = np.zeros((self.sequence_length, self.num_input_channels), dtype=np.float32)

        if segment_data.shape[0] != self.sequence_length or segment_data.shape[1] != self.num_input_channels:
            segment_data = pad_or_truncate(segment_data, self.sequence_length, self.num_input_channels)

        if self.is_training_set and self.augment_data:
            segment_data = self._apply_temporal_occlusion(segment_data)

        if self.scaler:
            segment_data = self.scaler.transform(segment_data)

        segment_data_transposed = segment_data.T
        sequence_tensor = torch.from_numpy(segment_data_transposed).float()
        
        label = self.task_labels[idx]
        # For CrossEntropyLoss, target should be LongTensor for class indices
        # For BCEWithLogitsLoss, target should be FloatTensor (already is for binary)
        if self.task_name in ['play_years', 'level']:
            label_tensor = torch.tensor(label, dtype=torch.long)
        else: # gender, handedness
            label_tensor = torch.tensor(label, dtype=torch.float).unsqueeze(0) # BCEWithLogitsLoss expects (N, *) or (N, C)

        return sequence_tensor, label_tensor

# --- Plotting and Metrics (similar to before, but adapted for single task) ---
def plot_task_confusion_matrix(y_true, y_pred_indices, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred_indices, labels=np.arange(len(classes))) # Ensure labels are 0 to N-1
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path); plt.close()
    else: plt.show(); plt.close()


# --- Main Training Function for a Single Task ---
def train_single_task(task_name, num_classes_for_task, train_df, valid_df, scaler, device):
    print(f"\n--- Training for Task: {task_name} ---")
    print(f"Number of output classes for this task: {num_classes_for_task}")

    # Define save paths
    task_model_save_dir = os.path.join(BASE_MODEL_SAVE_DIR, task_name) # e.g., trained_separate_models_v2/gender
    model_save_path = os.path.join(task_model_save_dir, f"cnn1d_{task_name}_best.pth") # Save best model for the task
    
    task_cm_save_dir = os.path.join(CONFUSION_MATRIX_BASE_DIR, task_name) # e.g., confusion_matrices_separate_v2/gender

    # --- 修改點：正確創建目錄 ---
    os.makedirs(task_model_save_dir, exist_ok=True) # Create directory for this task's models
    os.makedirs(task_cm_save_dir, exist_ok=True)   # Create directory for this task's CMs
    # -------------------------

    aug_params = {
        'occlusion_prob': OCCLUSION_PROBABILITY,
        'max_occlusion_len_ratio': MAX_OCCLUSION_LENGTH_RATIO,
        'occlusion_val': OCCLUSION_VALUE
    }
    train_dataset = TaskSpecificSegmentedDataset(train_df, SEQUENCE_LENGTH, INPUT_CHANNELS, scaler,
                                               task_name, is_training_set=True, augment_data=AUGMENT_TRAIN_DATA, **aug_params)
    valid_dataset = TaskSpecificSegmentedDataset(valid_df, SEQUENCE_LENGTH, INPUT_CHANNELS, scaler,
                                               task_name, is_training_set=False)
    
    current_batch_size = BATCH_SIZE
    if len(train_dataset) < BATCH_SIZE :
        current_batch_size = max(16, len(train_dataset) // 4 if len(train_dataset) // 4 > 0 else len(train_dataset))
        # print(f"Adjusted batch size for task {task_name} to {current_batch_size} due to small dataset size.")


    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=current_batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = BaseCNN1D(input_channels=INPUT_CHANNELS, num_classes_for_task=num_classes_for_task).to(device)
    
    is_multiclass = task_name in ['play_years', 'level']
    if is_multiclass:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=BASE_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.2, verbose=False)

    best_val_metric = -1.0
    
    epochs_to_run = BASE_NUM_EPOCHS // 2 if task_name in ['gender', 'handedness'] else BASE_NUM_EPOCHS

    for epoch in range(epochs_to_run):
        model.train()
        epoch_train_loss, train_corrects, train_total_for_acc = 0.0, 0, 0
        all_train_preds_for_f1, all_train_targets_for_f1 = [], []

        for sequences, targets_batch in train_loader: # Removed tqdm here for cleaner per-task output
            sequences, targets_batch = sequences.to(device), targets_batch.to(device)
            optimizer.zero_grad()
            logits_batch = model(sequences)

            if is_multiclass:
                loss = criterion(logits_batch, targets_batch)
                preds_indices = torch.argmax(logits_batch, dim=1)
                train_corrects += torch.sum(preds_indices == targets_batch).item()
                all_train_preds_for_f1.extend(preds_indices.cpu().numpy())
                all_train_targets_for_f1.extend(targets_batch.cpu().numpy())
            else:
                loss = criterion(logits_batch.squeeze(dim=1), targets_batch.squeeze(dim=1))
                preds_binary = (torch.sigmoid(logits_batch.squeeze(dim=1)) >= 0.5).float()
                train_corrects += torch.sum(preds_binary == targets_batch.squeeze(dim=1)).item()
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_total_for_acc += targets_batch.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_acc = train_corrects / train_total_for_acc if train_total_for_acc > 0 else 0.0
        train_f1 = f1_score(all_train_targets_for_f1, all_train_preds_for_f1, average='macro', zero_division=0) if is_multiclass and all_train_targets_for_f1 else 0.0

        model.eval()
        epoch_val_loss, val_corrects, val_total_for_acc = 0.0, 0, 0
        all_val_preds_for_cm, all_val_targets_for_cm = [], []

        with torch.no_grad():
            for sequences, targets_batch in valid_loader: # Removed tqdm here
                sequences, targets_batch = sequences.to(device), targets_batch.to(device)
                logits_batch = model(sequences)

                if is_multiclass:
                    loss = criterion(logits_batch, targets_batch)
                    preds_indices = torch.argmax(logits_batch, dim=1)
                    val_corrects += torch.sum(preds_indices == targets_batch).item()
                    all_val_preds_for_cm.extend(preds_indices.cpu().numpy())
                    all_val_targets_for_cm.extend(targets_batch.cpu().numpy())
                else:
                    loss = criterion(logits_batch.squeeze(dim=1), targets_batch.squeeze(dim=1))
                    preds_binary = (torch.sigmoid(logits_batch.squeeze(dim=1)) >= 0.5).float()
                    val_corrects += torch.sum(preds_binary == targets_batch.squeeze(dim=1)).item()
                    all_val_preds_for_cm.extend(preds_binary.cpu().numpy())
                    all_val_targets_for_cm.extend(targets_batch.squeeze(dim=1).cpu().numpy())

                epoch_val_loss += loss.item()
                val_total_for_acc += targets_batch.size(0)
        
        avg_val_loss = epoch_val_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0
        val_acc = val_corrects / val_total_for_acc if val_total_for_acc > 0 else 0.0
        val_f1 = f1_score(all_val_targets_for_cm, all_val_preds_for_cm, average='macro', zero_division=0) if is_multiclass and all_val_targets_for_cm else 0.0
        
        metric_to_print = f"Acc:{val_acc:.4f}"
        if is_multiclass: metric_to_print += f" F1:{val_f1:.4f}"
        
        print(f"Epoch {epoch+1}/{epochs_to_run} -> Task:{task_name} | Tr L:{avg_train_loss:.4f} Acc:{train_acc:.4f} {'F1:'+str(round(train_f1,4)) if is_multiclass else ''} | Val L:{avg_val_loss:.4f} {metric_to_print}")
        
        scheduler.step(avg_val_loss)
        current_metric_for_saving = val_f1 if is_multiclass else val_acc

        if current_metric_for_saving > best_val_metric:
            print(f"Val metric for {task_name} improved ({best_val_metric:.4f} -> {current_metric_for_saving:.4f}). Saving model...")
            best_val_metric = current_metric_for_saving
            torch.save(model.state_dict(), model_save_path) # model_save_path is now well-defined directory + file
            
            if task_name == 'gender': class_names = ['Female(2)', 'Male(1)']
            elif task_name == 'handedness': class_names = ['Left(2)', 'Right(1)']
            elif task_name == 'play_years': class_names = [f'PY_{i}' for i in range(3)]
            elif task_name == 'level': class_names = ['Lvl_甲(2)', 'Lvl_乙(3)', 'Lvl_青國(4)', 'Lvl_青選(5)']
            else: class_names = [str(i) for i in range(num_classes_for_task)]
            
            # Save CM to the task-specific CM directory
            plot_task_confusion_matrix(np.array(all_val_targets_for_cm), np.array(all_val_preds_for_cm),
                                       classes=class_names, title=f'Epoch {epoch+1} - {task_name} CM',
                                       save_path=os.path.join(task_cm_save_dir, f"best_epoch_cm_{task_name}.png"))

    print(f"Training for task {task_name} finished. Best Val Metric: {best_val_metric:.4f}")


# --- Main Execution Block ---
if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei', 'PingFang SC']
    matplotlib.rcParams['axes.unicode_minus'] = False
    multiprocessing.freeze_support()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Load Scaler (Fit once on all training data features) ---
    print("Fitting/Loading Scaler on ALL Segmented Training Data Features...")
    # This scaler will be used by all task-specific datasets.
    # It's fit on the input features X, not on labels.
    try:
        main_train_info_df = pd.read_csv(TRAIN_INFO_FILE_SEGMENTED)
    except FileNotFoundError: print(f"Error: {TRAIN_INFO_FILE_SEGMENTED} not found. Run preprocess."); exit()
    if main_train_info_df.empty: print(f"Error: {TRAIN_INFO_FILE_SEGMENTED} is empty."); exit()

    scaler_path_global = os.path.join(BASE_SCALER_SAVE_DIR, "global_scaler_all_features.joblib")
    os.makedirs(BASE_SCALER_SAVE_DIR, exist_ok=True)

    if os.path.exists(scaler_path_global):
        print(f"Loading existing global scaler from {scaler_path_global}")
        scaler = joblib.load(scaler_path_global)
    else:
        print("Fitting new global scaler...")
        scaler = StandardScaler()
        num_segments_to_sample = min(len(main_train_info_df), 300000) # Sample a good portion for scaler
        if num_segments_to_sample > 0:
            sampled_infos = main_train_info_df.sample(n=num_segments_to_sample, random_state=42)
            all_data_for_scaler = []
            for _, row in tqdm(sampled_infos.iterrows(), total=len(sampled_infos), desc="Loading segments for global scaler"):
                try:
                    segment_data = np.load(row['segment_filepath'])
                    if segment_data.shape == (SEQUENCE_LENGTH, INPUT_CHANNELS):
                        all_data_for_scaler.append(segment_data)
                except Exception as e: print(f"Scaler: Error loading {row['segment_filepath']}: {e}")
            if all_data_for_scaler:
                concatenated_data = np.concatenate(all_data_for_scaler, axis=0) # (N_total_timesteps, 7)
                scaler.fit(concatenated_data)
                print(f"Global Scaler Mean: {scaler.mean_[:3]}...")
                joblib.dump(scaler, scaler_path_global)
                print(f"Global Scaler saved to {scaler_path_global}")
            else: print("Error: No data for global scaler."); exit()
        else: print("Warning: No segments to sample for global scaler."); exit()

    # --- Load Full Datasets (for passing to training functions) ---
    full_train_df = pd.read_csv(TRAIN_INFO_FILE_SEGMENTED)
    full_valid_df = pd.read_csv(VALID_INFO_FILE_SEGMENTED)
    if full_train_df.empty or full_valid_df.empty:
        print("Error: Full train or valid segmented info is empty."); exit()

    # --- Define Tasks and Train ---
    tasks = {
        'gender': {'num_classes': 1, 'class_names': ['Female(2)', 'Male(1)']}, # Output 1 logit for BCE
        'handedness': {'num_classes': 1, 'class_names': ['Left(2)', 'Right(1)']}, # Output 1 logit
        'play_years': {'num_classes': 3, 'class_names': ['PY_0', 'PY_1', 'PY_2']}, # Output 3 logits for CE
        'level': {'num_classes': 4, 'class_names': ['Lvl_甲(2)', 'Lvl_乙(3)', 'Lvl_青國(4)', 'Lvl_青選(5)']} # Output 4 logits
    }

    for task_name, task_config in tasks.items():
        train_single_task(
            task_name=task_name,
            num_classes_for_task=task_config['num_classes'],
            train_df=full_train_df.copy(), # Pass copies to avoid accidental modification
            valid_df=full_valid_df.copy(),
            scaler=scaler, # Use the globally fitted scaler
            device=DEVICE
        )
    
    print("All tasks trained.")
# --- END OF FILE train_multitask_separate_models.py ---
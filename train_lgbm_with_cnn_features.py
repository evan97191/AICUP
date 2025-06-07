# --- START OF FILE train_lgbm_with_cnn_features.py ---
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split # Not strictly needed if using pre-defined valid set
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
import joblib
from tqdm import tqdm

# --- Configuration ---
FEATURE_DIR = "lgbm_features_from_multi_cnn" # <-- CHANGED: Directory with concatenated features
TRAIN_FEATURES_FILE = os.path.join(FEATURE_DIR, "train_multi_cnn_features.csv") # <-- CHANGED
VALID_FEATURES_FILE = os.path.join(FEATURE_DIR, "valid_multi_cnn_features.csv") # <-- CHANGED

# Original segmented info files to get labels (remains same)
TRAIN_INFO_SEGMENTED_ORIGINAL = os.path.join("segmented_data_final", "train_info_segmented.csv")
VALID_INFO_SEGMENTED_ORIGINAL = os.path.join("segmented_data_final", "valid_info_segmented.csv")

LGBM_MODEL_OUTPUT_DIR = "lgbm_models_from_multi_cnn" # <-- CHANGED: New output for these LGBM models
os.makedirs(LGBM_MODEL_OUTPUT_DIR, exist_ok=True)

# LightGBM Parameters (can be tuned)
LGBM_GENERAL_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 40, # Might need more leaves for more features
    'learning_rate': 0.03,
    'feature_fraction': 0.8, # Consider a bit lower if many features are correlated
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'lambda_l1': 0.1, # Add some L1 regularization
    'lambda_l2': 0.1, # Add some L2 regularization
    'min_child_samples': 20, # Default
    'verbose': -1,
    'n_estimators': 1500,
    'n_jobs': -1,
    'seed': 42,
}

# Task definitions (remains same)
TASKS_LGBM = {
    'gender': {'label_col': 'gender', 'objective': 'binary', 'metric': ['binary_logloss', 'auc'], 'num_class': 1, 'type': 'binary'},
    'handedness': {'label_col': 'hold racket handed', 'objective': 'binary', 'metric': ['binary_logloss', 'auc'], 'num_class': 1, 'type': 'binary'},
    'play_years': {'label_col': 'play years', 'objective': 'multiclass', 'metric': ['multi_logloss', 'multi_error'], 'num_class': 3, 'type': 'multiclass'},
    'level': {'label_col': 'level', 'objective': 'multiclass', 'metric': ['multi_logloss', 'multi_error'], 'num_class': 4, 'type': 'multiclass'}
}
LEVEL_MAP_LGBM = {2: 0, 3: 1, 4: 2, 5: 3}

# train_lgbm_for_task function (remains mostly the same, just ensure it uses the correct feature columns)
def train_lgbm_for_task(task_name, task_config, df_train_features, df_valid_features,
                        df_train_labels_info, df_valid_labels_info):
    print(f"\n--- Training LightGBM for Task: {task_name} (using Multi-CNN features) ---")

    # Merge features with labels (make sure 'segment_id' is in df_train_features correctly)
    df_train = pd.merge(df_train_features, df_train_labels_info[['segment_id', task_config['label_col']]], on='segment_id', how='left')
    df_valid = pd.merge(df_valid_features, df_valid_labels_info[['segment_id', task_config['label_col']]], on='segment_id', how='left')
    
    df_train.dropna(subset=[task_config['label_col']], inplace=True)
    df_valid.dropna(subset=[task_config['label_col']], inplace=True)

    if df_train.empty or df_valid.empty:
        print(f"Not enough data for task {task_name} after merging. Skipping.")
        return None

    # Select ALL feature columns (now they are named like 'gender_cnn_feat_0', 'level_cnn_feat_63', etc.)
    feature_cols_train = [col for col in df_train.columns if '_cnn_feat_' in col]
    feature_cols_valid = [col for col in df_valid.columns if '_cnn_feat_' in col]

    if not feature_cols_train or not feature_cols_valid :
        print(f"Error: No feature columns found for task {task_name}. Check feature file generation.")
        return None

    X_train = df_train[feature_cols_train].values
    X_valid = df_valid[feature_cols_valid].values
    
    y_train_raw = df_train[task_config['label_col']]
    y_valid_raw = df_valid[task_config['label_col']]

    # Prepare labels (same as before)
    if task_name == 'level':
        y_train = y_train_raw.map(LEVEL_MAP_LGBM).astype(int).values
        y_valid = y_valid_raw.map(LEVEL_MAP_LGBM).astype(int).values
    elif task_name == 'play_years':
        y_train = y_train_raw.astype(int).values
        y_valid = y_valid_raw.astype(int).values
    elif task_name == 'gender':
        y_train = y_train_raw.apply(lambda x: 1 if x == 1 else 0).astype(int).values
        y_valid = y_valid_raw.apply(lambda x: 1 if x == 1 else 0).astype(int).values
    elif task_name == 'handedness':
        y_train = y_train_raw.apply(lambda x: 1 if x == 1 else 0).astype(int).values
        y_valid = y_valid_raw.apply(lambda x: 1 if x == 1 else 0).astype(int).values
    else: return None

    if X_train.shape[0] == 0 or X_valid.shape[0] == 0: print(f"No samples for {task_name}. Skip."); return None

    lgbm_params_task = LGBM_GENERAL_PARAMS.copy()
    lgbm_params_task['objective'] = task_config['objective']
    lgbm_params_task['metric'] = task_config['metric']
    if task_config['objective'] == 'multiclass':
        lgbm_params_task['num_class'] = task_config['num_class']

    model = lgb.LGBMClassifier(**lgbm_params_task)
    print(f"Training LGBM model for {task_name} with {X_train.shape[1]} features...")
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric=task_config['metric'],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=300)]) # Longer patience

    # Evaluation (same as before)
    if task_config['type'] == 'binary':
        y_pred_proba_valid = model.predict_proba(X_valid)[:, 1]
        y_pred_valid = (y_pred_proba_valid >= 0.5).astype(int)
        acc = accuracy_score(y_valid, y_pred_valid)
        auc = roc_auc_score(y_valid, y_pred_proba_valid)
        f1 = f1_score(y_valid, y_pred_valid, zero_division=0)
        print(f"Validation - Task: {task_name}, Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
    else: # Multiclass
        y_pred_proba_valid = model.predict_proba(X_valid)
        y_pred_valid = np.argmax(y_pred_proba_valid, axis=1)
        acc = accuracy_score(y_valid, y_pred_valid)
        f1 = f1_score(y_valid, y_pred_valid, average='macro', zero_division=0)
        print(f"Validation - Task: {task_name}, Acc: {acc:.4f}, Macro F1: {f1:.4f}")

    model_path = os.path.join(LGBM_MODEL_OUTPUT_DIR, f"lgbm_multi_cnn_feat_{task_name}.txt")
    model.booster_.save_model(model_path)
    print(f"LGBM model for {task_name} (multi-cnn feat) saved to {model_path}")
    return model

if __name__ == '__main__':
    # ... (loading df_train_features, df_valid_features, df_train_labels_info, df_valid_labels_info) ...
    # The rest of the __main__ block is the same as your previous LGBM training script.
    print("Loading Multi-CNN features for LightGBM training...")
    try:
        df_train_features = pd.read_csv(TRAIN_FEATURES_FILE)
        df_valid_features = pd.read_csv(VALID_FEATURES_FILE)
    except FileNotFoundError as e:
        print(f"Error: Multi-CNN feature file not found: {e}. Run extract_cnn_features_multi_model.py first."); exit()

    print("Loading original segmented info for labels...")
    try:
        df_train_labels_info = pd.read_csv(TRAIN_INFO_SEGMENTED_ORIGINAL, dtype={'original_unique_id': str, 'segment_id': str})
        df_valid_labels_info = pd.read_csv(VALID_INFO_SEGMENTED_ORIGINAL, dtype={'original_unique_id': str, 'segment_id': str})
    except FileNotFoundError as e:
        print(f"Error: Original segmented info file not found: {e}. Ensure preprocess_data.py ran."); exit()

    trained_lgbm_models = {}
    for task_name, config in TASKS_LGBM.items():
        model = train_lgbm_for_task(task_name, config,
                                    df_train_features.copy(), df_valid_features.copy(), # Pass copies
                                    df_train_labels_info, df_valid_labels_info)
        if model:
            trained_lgbm_models[task_name] = model
            
    print("\nAll LightGBM models (from multi-CNN features) trained.")
# --- END OF FILE train_lgbm_with_multi_cnn_features.py ---
# --- START OF FILE model.py (部分修改) ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN1D(nn.Module):
    def __init__(self, input_channels=7, num_classes_for_task=1, sequence_length=None): # num_classes_for_task 可能是最後的輸出頭
        super(BaseCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.relu1 = nn.ReLU(inplace=True) # Giving unique names can help if hooking specific layers
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        # This is the layer whose output we likely want as features for LightGBM
        self.feature_extractor_fc = nn.Linear(in_features=128, out_features=64)
        self.feature_extractor_relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5) # Dropout is usually only active during training
        
        # Final classification head (not used when extracting features for LightGBM)
        self.fc_logits = nn.Linear(in_features=64, out_features=num_classes_for_task)

    def forward(self, x, extract_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        
        # Features for LightGBM are typically taken from here
        features = self.feature_extractor_fc(x)
        features = self.feature_extractor_relu(features)
        # Note: Dropout is applied after this if extract_features=False.
        # For feature extraction, we usually want the output *before* dropout,
        # or ensure dropout is turned off (model.eval()).

        if extract_features:
            return features # Return the 64-dimensional feature vector

        x_after_dropout = self.dropout(features) # Apply dropout only if not extracting features for LGBM
        logits = self.fc_logits(x_after_dropout)
        return logits

    # get_predictions_from_logits (if you had one, might not be directly used by LGBM training)
    # ...
# --- END OF FILE model.py ---
# Root script: main.py

from data.preprocess import preprocess_dataset
from features.mel_spectrogram import generate_mel_spectrogram
from features.cwt_transform import generate_cwt_features
from features.handcrafted_acoustic import extract_handcrafted_features
from models.mobilenet_v3 import MobileNetV3
from models.efficientnet_b7 import EfficientNetB7
from models.linformer import LinformerEncoder
from models.performer import PerformerEncoder
from models.fusion_attention import AttentionFusion
from optimization.xgboost_classifier import XGBoostClassifier
from optimization.bohb_tuner import tune_hyperparameters
from interpretability.shap_analysis import explain_with_shap
from utils.metrics import evaluate_model
import numpy as np

# Step 1: Preprocess and augment dataset
train_data, val_data, test_data = preprocess_dataset("./data/", augment=True, k_fold=5)

# Step 2: Generate Mel-Spectrogram and CWT features
mel_train = generate_mel_spectrogram(train_data)
cwt_train = generate_cwt_features(train_data)

# Step 3: Extract handcrafted acoustic features
stat_features_train = extract_handcrafted_features(train_data)

# Step 4: CNN and ViT feature extraction
mobilenet_feats = MobileNetV3().extract(mel_train)
efficientnet_feats = EfficientNetB7().extract(mel_train)
linformer_feats = LinformerEncoder().extract(mel_train)
performer_feats = PerformerEncoder().extract(mel_train)

# Step 5: Attention-based feature fusion
fusion_module = AttentionFusion()
fused_features = fusion_module.combine([
    mobilenet_feats,
    efficientnet_feats,
    linformer_feats,
    performer_feats,
    stat_features_train
])

# Step 6: Hyperparameter tuning and classification
best_params = tune_hyperparameters(fused_features, train_data.labels)
xgb_model = XGBoostClassifier(params=best_params)
xgb_model.train(fused_features, train_data.labels)

# Step 7: Evaluation
predictions = xgb_model.predict(test_data.features)
evaluate_model(test_data.labels, predictions)

# Step 8: SHAP interpretability
explain_with_shap(xgb_model, fused_features, train_data.labels)

"""
Project Directory Structure:
speech-disorder-detection/
??? README.md
??? requirements.txt
??? config/
?   ??? config.yaml
??? data/
?   ??? preprocess.py
??? features/
?   ??? mel_spectrogram.py
?   ??? cwt_transform.py
?   ??? handcrafted_acoustic.py
??? models/
?   ??? mobilenet_v3.py
?   ??? efficientnet_b7.py
?   ??? linformer.py
?   ??? performer.py
?   ??? fusion_attention.py
??? training/
?   ??? train.py
?   ??? evaluate.py
??? optimization/
?   ??? bohb_tuner.py
?   ??? xgboost_classifier.py
??? interpretability/
?   ??? shap_analysis.py
??? utils/
    ??? logger.py
    ??? metrics.py
"""

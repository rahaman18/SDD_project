# Multi-Feature Fusion-Based Speech Disorder Classification Using MobileNetV3-EfficientNetB7, Linformer-Performer, and SHAP-Aware XGBoost
#Multi-Feature Fusion-Based Speech Disorder Classification Using MobileNetV3-EfficientNetB7, Linformer-Performer, and SHAP-Aware XGBoost


This repository presents a complete pipeline for classifying healthy and pathological speech using a hybrid deep learning approach that combines CNNs (MobileNetV3, EfficientNetB7), Vision Transformers (Linformer, Performer), handcrafted features, and interpretable machine learning via SHAP values. The model is optimized using Bayesian Optimization with Hyperband (BOHB) and classified using a fine-tuned XGBoost model.

 Key Features

- **Mel-Spectrogram and CWT Generation** for high-resolution time-frequency features
- **Hybrid Feature Extraction**:
  - CNNs: MobileNetV3 & EfficientNetB7
  - Transformers: Linformer & Performer
  - Statistical & Acoustic handcrafted features (MFCCs, jitter, shimmer, etc.)
- **Attention-Based Feature Fusion** to emphasize discriminative features
- **XGBoost Classifier with BOHB Optimization** for robust performance
- **SHAP Values** for interpretable model decisions
- **Stratified 5-Fold Cross Validation** to prevent data leakage and ensure generalizability

Project Structure

```
speech-disorder-detection/
 data/                  # Raw audio & preprocessing scripts
 features/              # Mel, CWT & handcrafted feature extractors
 models/                # CNN and ViT-based encoders + fusion module
 training/              # Training and evaluation scripts
 optimization/          # BOHB tuner and XGBoost integration
interpretability/      # SHAP analysis and plots
 utils/                 # Metrics, logger, helper functions
 main.py                # Full training pipeline entry point
 requirements.txt       # Dependencies
 README.md              # Project overview
```

 Requirements

```bash
pip install -r requirements.txt
```

Dependencies include:
- `tensorflow`, `torch`, `librosa`, `xgboost`, `shap`, `hyperopt`, `audiomentations`, `pywt`

Getting Started

1. Place `.wav` files in `data/` directory and ensure filenames contain labels (e.g., `healthy_x.wav`, `pathological_y.wav`).
2. Run the main pipeline:

```bash
python main.py
```

3. Visualize SHAP explanations and evaluate performance using the output metrics.

 Output Metrics
- Accuracy, Precision, Recall, F1-score
- Confidence Interval, Standard Deviation
- SHAP feature importance plots
- AUROC and AUPRC curves

## Notes
- Ensure input spectrograms are normalized and resized to 128x128.
- Preprocessing includes augmentation (pitch shift, noise, time stretch) for robustness.
- Data split follows stratified K-fold to preserve class balance.

##  Contributing
Pull requests and research contributions are welcome! Please cite the original authors and datasets (SVD, VOICED) when using this repository.

##  License
MIT License


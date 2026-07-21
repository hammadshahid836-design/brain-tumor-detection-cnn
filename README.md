# Brain Tumor Detection — CNN Classifier

A convolutional neural network trained from scratch to classify brain MRI scans as **tumor present** or **no tumor**, built around a small (253-image) dataset. The whole pipeline — data splitting, augmentation, architecture, training, and evaluation — is designed specifically around the constraints of limited medical imaging data.

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **79%** |
| Test ROC-AUC | **0.81** |
| Recall — tumor ("yes") | 92% |
| Recall — no tumor ("no") | 54% |

![Training curves](figures/training_curves.png)
![Confusion matrix](figures/confusion_matrix.png)

## 📁 Project Structure

```
brain-tumor-detection-cnn/
├── notebooks/
│   └── brain_mri_cnn_classifier.ipynb   # Full pipeline: EDA → training → evaluation
├── models/
│   ├── brain_mri_cnn.keras              # Trained model weights
│   └── final_results.json               # Test set metrics
├── figures/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

## 🧠 Dataset

253 brain MRI images across 2 classes:
- **yes** — 155 images (tumor present)
- **no** — 98 images (no tumor)

Split **70% train / 15% val / 15% test**, stratified by class. Images are resized to 150×150 and converted to RGB.

> The dataset isn't included in this repo. Point the notebook's `DATA_DIR` variable at your own copy, organized as `yes/` and `no/` subfolders of image files.

## 🏗️ Model Design

With only 177 training images, avoiding overfitting was the central design constraint — not squeezing out more raw capacity:

- **Small filter counts** (16 → 32 → 64) instead of typical larger CNNs
- **`SpatialDropout2D`** after each conv block — drops entire feature maps rather than individual pixels, which regularizes convolutional layers more effectively than standard dropout
- **`GlobalAveragePooling2D`** instead of `Flatten` + large Dense layer — massively reduces parameter count and overfitting risk
- **L2 weight regularization** on both conv and dense layers
- **Heavy data augmentation** (rotation, shift, shear, zoom, flip, brightness) to synthetically expand the effective training set
- **Class weighting** to correct the mild "yes"/"no" imbalance
- **Validation AUC** (not accuracy) used for early stopping and LR scheduling — more informative on a small, imbalanced validation set

Total model size: **~26K parameters** — intentionally compact.

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/brain-tumor-detection-cnn.git
   cd brain-tumor-detection-cnn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your dataset (folders `yes/` and `no/` of MRI images), and update `DATA_DIR` in the notebook.

4. Launch Jupyter and run the notebook:
   ```bash
   jupyter notebook notebooks/brain_mri_cnn_classifier.ipynb
   ```

## 🔮 Future Improvements

- **Transfer learning** — fine-tuning a pretrained ImageNet backbone (MobileNetV2, ResNet, EfficientNet) typically gives a strong boost on small medical imaging datasets. This wasn't used here due to no internet access to pretrained weights in the training environment, but is the natural next step.
- **More data**, especially "no tumor" examples — the class imbalance and small "no" sample size are likely the main driver of the asymmetric recall (92% vs 54%).
- **Grad-CAM visualization** — to inspect what regions of the MRI the model is actually attending to, which matters a lot for trust in a medical imaging context.

## ⚠️ Disclaimer

This model is trained on a small, academic dataset for **educational purposes only**. It is **not validated for clinical or diagnostic use** and should never be used to make real medical decisions.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

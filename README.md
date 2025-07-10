# 🖼️ Steganography Detector

Detects hidden data in images using machine learning.

## 📦 Features
- Extracts color, statistical & frequency features.
- Trains a Random Forest classifier to detect stego images.
- Streamlit app: upload image, see prediction, feature importance & confusion matrix.
- Visual explanations using SHAP.

## 🚀 Quick start
```bash
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run streamlit_app.py

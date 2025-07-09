# steganography-detector
# 🖼️ Steganography Detector

Detects hidden data in images using ML features like color histograms, edge maps, compression ratio, and DCT energy.

## 🚀 Features
- Upload JPG/PNG images via Streamlit
- Extract and visualize key image features
- Predict if image is "Stego" or "Clean"
- Download CSV prediction report

## 📦 Tech Stack
- Python, OpenCV, Scikit-learn, Streamlit
- RandomForest for classification

## 🧠 How it Works
Features extracted:
- Color Histogram
- Edge Density
- Compression Ratio
- DCT Energy

## 🖥️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/steganography-detector.git
cd steganography-detector
pip install -r requirements.txt
streamlit run app/stego_app.py

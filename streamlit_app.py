# 📂 File: stego_app.py

import os
import cv2
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import json
from scipy.fftpack import dct
from PIL import Image
import io

# ---------------------- Feature Extraction ----------------------
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    features = {}

    features['file_size'] = os.path.getsize(img_path)
    features['width'] = img.shape[1]
    features['height'] = img.shape[0]
    features['channels'] = img.shape[2]
    features['mean'] = img.mean()
    features['std'] = img.std()

    # Color histogram
    chans = cv2.split(img)
    colors = ('b', 'g', 'r')
    for i, chan in enumerate(chans):
        hist = cv2.calcHist([chan], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features[f'hist_mean_{colors[i]}'] = np.mean(hist)

    # Compression ratio
    pil_img = Image.open(img_path)
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=50)
    compressed_size = len(buf.getvalue())
    features['compression_ratio'] = features['file_size'] / compressed_size

    # DCT energy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct_trans = dct(dct(gray.astype(float), axis=0), axis=1)
    features['dct_energy'] = np.sum(np.square(dct_trans)) / (features['width'] * features['height'])

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (features['width'] * features['height'])
    features['edge_density'] = edge_density

    return pd.DataFrame([features])

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Steganography Detector", page_icon="🖼️")
st.title("🖼️ Steganography Detector")

st.markdown("This app detects whether an image may contain hidden data using machine learning. Upload an image and view detailed analysis below.")

# Upload
uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.read())

    features = extract_features("temp.jpg")

    if features is not None:
        with open("stego_model.pkl", "rb") as f:
            model = pickle.load(f)

        proba = model.predict_proba(features)
        prediction = model.predict(features)[0]
        label = "🔴 Possibly Stego" if prediction == 1 else "🟢 Clean"

        tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🧠 Feature Importance", "📷 Image Analysis"])

        with tab1:
            st.image("temp.jpg", caption="Uploaded Image", width=300)
            st.markdown(f"## Prediction: {label}")
            st.metric("Confidence", f"{proba[0][prediction]:.2f}")

            report = features.copy()
            report['prediction'] = label
            csv = report.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="report.csv">📥 Download Prediction Report</a>'
            st.markdown(href, unsafe_allow_html=True)

        with tab2:
            try:
                with open("top_features.json") as f:
                    top_feats = json.load(f)
                st.markdown("#### 🔍 Top Contributing Features:")
                for feat, val in top_feats:
                    st.write(f"- **{feat}** → `{val:.4f}`")
            except:
                st.info("📉 Feature importance file not found or not generated.")

        with tab3:
            st.subheader("Edge Map")
            edges = cv2.Canny(cv2.imread("temp.jpg"), 100, 200)
            st.image(edges, caption="Edge Detection", clamp=True)

            st.subheader("Color Histogram")
            fig, ax = plt.subplots()
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([cv2.imread("temp.jpg")], [i], None, [64], [0, 256])
                ax.plot(hist, color=col)
            st.pyplot(fig)

            with st.expander("📑 See Raw Image Features"):
                st.dataframe(features.T)

    else:
        st.warning("⚠️ Could not read the image.")

st.markdown("---")
st.markdown("🔐 Built with ❤️ by [Priyadarsini KM](mailto:priyadarsinikm04@gmail.com)")
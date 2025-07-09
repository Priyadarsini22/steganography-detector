
# extract_features.py
import os
import cv2
import numpy as np
from scipy.fftpack import dct
from PIL import Image
import io
import pandas as pd

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    features = {}

    # Basic features
    features['file_size'] = os.path.getsize(img_path)
    features['width'] = img.shape[1]
    features['height'] = img.shape[0]
    features['channels'] = img.shape[2]
    features['mean'] = img.mean()
    features['std'] = img.std()

    # Color histogram (mean of each channel)
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
    features['edge_density'] = np.sum(edges > 0) / (features['width'] * features['height'])

    return pd.DataFrame([features])

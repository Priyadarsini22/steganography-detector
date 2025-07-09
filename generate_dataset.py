# generate_dataset.py
import os
import pandas as pd
from extract_features import extract_features

data = []

# Clean images
for fname in os.listdir("images/clean"):
    path = f"images/clean/{fname}"
    feat = extract_features(path)
    if feat is not None:
        feat['label'] = 'clean'
        data.append(feat)

# Stego images
for fname in os.listdir("images/stego"):
    path = f"images/stego/{fname}"
    feat = extract_features(path)
    if feat is not None:
        feat['label'] = 'stego'
        data.append(feat)

# Combine and save
df = pd.concat(data, ignore_index=True)
df.to_csv("stego_features.csv", index=False)
print("✅ stego_features.csv created.")

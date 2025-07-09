# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import json

df = pd.read_csv("stego_features.csv")
X = df.drop("label", axis=1)
y = LabelEncoder().fit_transform(df["label"])

if len(X) < 2:
    print("❌ Not enough data to train.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("stego_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved to stego_model.pkl")

# Save top features
importances = model.feature_importances_
top_feats = sorted(zip(X.columns, importances), key=lambda x: -x[1])[:5]
with open("top_features.json", "w") as f:
    json.dump(top_feats, f)

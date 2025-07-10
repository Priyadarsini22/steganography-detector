import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import json
import matplotlib.pyplot as plt
import shap

# Load dataset
df = pd.read_csv("stego_features.csv")
X = df.drop("label", axis=1)
y = LabelEncoder().fit_transform(df["label"])

if len(X) < 2:
    print("❌ Not enough data to train.")
    exit()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
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
print("✅ Top features saved to top_features.json")

# Confusion matrix
cm = confusion_matrix(y_test, model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clean', 'Stego'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
print("✅ Confusion matrix saved to confusion_matrix.png")

# SHAP feature importance
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()
print("✅ SHAP summary plot saved to shap_summary.png")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from cnn_model import ConvNN

# Configuration
SLICE_LEN = 4
TARGET_FEATS = 18
NUM_CLASSES = 3
LABEL_MAP = {'mmtc': 0, 'embb': 1, 'urllc': 2}
LABEL_NAMES = ['mmtc', 'embb', 'urllc']

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_dataframe(df, scaler=None, fit_scaler=False):
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    X_df = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]

    y_flat = y_raw.map(LABEL_MAP).fillna(0).values.astype(int)
    X_flat = X_df.values

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X_flat = scaler.fit_transform(X_flat)
    else:
        X_flat = scaler.transform(X_flat)

    num_samples = len(X_flat)
    true_feats = X_flat.shape[1]

    if true_feats > TARGET_FEATS:
        X_flat = X_flat[:, :TARGET_FEATS]
    elif true_feats < TARGET_FEATS:
        pad = np.zeros((num_samples, TARGET_FEATS - true_feats))
        X_flat = np.hstack((X_flat, pad))

    pad_len = (SLICE_LEN - (num_samples % SLICE_LEN)) % SLICE_LEN
    if pad_len > 0:
        pad_X = np.zeros((pad_len, TARGET_FEATS))
        pad_y = np.zeros(pad_len, dtype=int)
        X_flat = np.concatenate([X_flat, pad_X], axis=0)
        y_flat = np.concatenate([y_flat, pad_y], axis=0)

    X = torch.tensor(X_flat, dtype=torch.float32).view(-1, SLICE_LEN, TARGET_FEATS)
    y = torch.tensor(y_flat, dtype=torch.long).view(-1, SLICE_LEN)[:, -1]

    return X, y, scaler


# -----------------------------
# Load datasets
# -----------------------------
train_files = [
    "data/client_0/round_1.csv",
    "data/client_1/round_1.csv",
    "data/client_2/round_1.csv"
]

train_dfs = [pd.read_csv(f) for f in train_files]
train_df = pd.concat(train_dfs, ignore_index=True)

test_df = pd.read_csv("data/server/test.csv")

# Fit scaler on training data
_, _, scaler = preprocess_dataframe(train_df, scaler=None, fit_scaler=True)

# Preprocess test data
X_test, y_test, _ = preprocess_dataframe(test_df, scaler=scaler, fit_scaler=False)

# -----------------------------
# Load trained model
# -----------------------------
model = ConvNN(slice_len=SLICE_LEN, num_feats=TARGET_FEATS, classes=NUM_CLASSES)
model.load_state_dict(torch.load("centralized_baseline_model.pt"))
model.eval()

# -----------------------------
# Predictions
# -----------------------------
with torch.no_grad():
    outputs = model(X_test)
    _, preds = torch.max(outputs, 1)

y_true = y_test.numpy()
y_pred = preds.numpy()

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Centralized Baseline Confusion Matrix")

plt.tight_layout()
plt.savefig("centralized_confusion_matrix.png")

print("Saved figure: centralized_confusion_matrix.png")
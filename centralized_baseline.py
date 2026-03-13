
import csv
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from cnn_model import ConvNN

# -----------------------------
# Configuration
# -----------------------------
SLICE_LEN = 4
TARGET_FEATS = 18
NUM_CLASSES = 3
EPOCHS = 10
LR = 0.001

LABEL_MAP = {'mmtc': 0, 'embb': 1, 'urllc': 2}

# -----------------------------
# Helper function
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

    # Pad/truncate features to TARGET_FEATS
    num_samples = len(X_flat)
    true_feats = X_flat.shape[1]

    if true_feats > TARGET_FEATS:
        X_flat = X_flat[:, :TARGET_FEATS]
    elif true_feats < TARGET_FEATS:
        pad = np.zeros((num_samples, TARGET_FEATS - true_feats))
        X_flat = np.hstack((X_flat, pad))

    # Pad rows so total rows is divisible by SLICE_LEN
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
# Load centralized training data
# -----------------------------
train_files = [
    "data/client_0/round_1.csv",
    "data/client_1/round_1.csv",
    "data/client_2/round_1.csv"
]

train_dfs = [pd.read_csv(f) for f in train_files]
train_df = pd.concat(train_dfs, ignore_index=True)

test_df = pd.read_csv("data/server/test.csv")

print(f"Centralized training samples (raw rows): {len(train_df)}")
print(f"Test samples (raw rows): {len(test_df)}")

# -----------------------------
# Preprocess
# -----------------------------
X_train, y_train, scaler = preprocess_dataframe(train_df, scaler=None, fit_scaler=True)
X_test, y_test, _ = preprocess_dataframe(test_df, scaler=scaler, fit_scaler=False)

print(f"Training tensor shape: {X_train.shape}")
print(f"Test tensor shape: {X_test.shape}")

# -----------------------------
# Model, loss, optimizer
# -----------------------------
model = ConvNN(slice_len=SLICE_LEN, num_feats=TARGET_FEATS, classes=NUM_CLASSES)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training loop
# -----------------------------

metrics = []


for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Training accuracy
    _, train_pred = torch.max(outputs.data, 1)
    train_acc = (train_pred == y_train).sum().item() / len(y_train) * 100

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        _, test_pred = torch.max(test_outputs.data, 1)
        test_acc = (test_pred == y_test).sum().item() / len(y_test) * 100

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
    )


    metrics.append({
        "epoch": epoch,
        "train_loss": loss.item(),
        "train_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc
    })

pd.DataFrame(metrics).to_csv("centralized_baseline_metrics.csv", index=False)
print("Saved metrics: centralized_baseline_metrics.csv")

# -----------------------------
# Save final model
# -----------------------------
torch.save(model.state_dict(), "centralized_baseline_model.pt")

print("Saved model: centralized_baseline_model.pt")

print("Saved model: centralized_baseline_model.pt")


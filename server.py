import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time  # <--- NEW
from sklearn.preprocessing import StandardScaler
from cnn_model import ConvNN 
from typing import Dict, Optional, Tuple

history = []

def load_test_data():
    print("Loading Global Test Set...")
    path = "/app/data/server/test.csv"
    if not os.path.exists(path): return None, None
    
    df = pd.read_csv(path)
    if 'Timestamp' in df.columns: df = df.drop(columns=['Timestamp'])
    
    label_map = {'mmtc': 0, 'embb': 1, 'urllc': 2}
    scaler = StandardScaler()
    slice_len = 4
    
    X_df = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]
    y_flat = y_raw.map(label_map).fillna(0).values.astype(int)
    X_flat = scaler.fit_transform(X_df.values)
    
    target_feats = 18
    if X_flat.shape[1] > target_feats: X_flat = X_flat[:, :target_feats]
    elif X_flat.shape[1] < target_feats:
        pad = np.zeros((len(X_flat), target_feats - X_flat.shape[1]))
        X_flat = np.hstack((X_flat, pad))

    pad_len = (slice_len - (len(X_flat) % slice_len)) % slice_len
    if pad_len > 0:
        X_flat = np.concatenate([X_flat, np.zeros((pad_len, target_feats))], axis=0)
        y_flat = np.concatenate([y_flat, np.zeros(pad_len, dtype=int)], axis=0)

    X = torch.tensor(X_flat, dtype=torch.float32).view(-1, slice_len, target_feats)
    y = torch.tensor(y_flat, dtype=torch.long).view(-1, slice_len)[:, -1]
    return X, y

X_test, y_test = load_test_data()

def fit_config(server_round: int):
    return {"server_round": server_round}

def get_evaluate_fn(model):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        if X_test is None: return None

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        model.eval()
        criterion = nn.NLLLoss()
        with torch.no_grad():
            out = model(X_test)
            loss = criterion(out, y_test).item()
            _, predicted = torch.max(out.data, 1)
            correct = (predicted == y_test).sum().item()
            accuracy = (correct / len(y_test)) * 100

        print(f"\n[ROUND {server_round}] GLOBAL ACCURACY: {accuracy:.2f}% | LOSS: {loss:.4f}")
        
        history.append({'round': server_round, 'global_accuracy': accuracy, 'global_loss': loss})
        pd.DataFrame(history).to_csv("/app/server_metrics.csv", index=False)
        torch.save(model.state_dict(), f"/app/global_model.pt")

        return loss, {"accuracy": accuracy}

    return evaluate

if __name__ == "__main__":
    model = ConvNN(slice_len=4, num_feats=18, classes=3)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )
    
    # --- SLEEP TO ALLOW COPYING ---
    print("Server finished. Sleeping for 1 hour to allow file download...")
    time.sleep(3600)

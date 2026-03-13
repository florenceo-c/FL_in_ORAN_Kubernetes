import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
import json
import time
from sklearn.preprocessing import StandardScaler
from cnn_model import ConvNN 

class CsvClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.net = None
        self.label_map = {'mmtc': 0, 'embb': 1, 'urllc': 2}
        self.history = [] 
        self.scaler = StandardScaler()
        self.slice_len = 4 

    def get_parameters(self, config):
        dummy = ConvNN(slice_len=self.slice_len, num_feats=18, classes=3)
        return [val.cpu().numpy() for _, val in dummy.state_dict().items()]

    def calculate_accuracy(self, X, y):
        with torch.no_grad():
            outputs = self.net(X)
            _, predicted = torch.max(outputs.data, 1)
            total = y.size(0)
            correct = (predicted == y).sum().item()
            if total == 0: return 0.0
            return (correct / total) * 100

    def fit(self, parameters, config):
        rnd = config.get("server_round", 1)
        path = f"/app/data/client_{self.client_id}/round_{rnd}.csv"

        if not os.path.exists(path):
            return parameters, 0, {}

        try:
            # 1. Load & Prep Data
            df = pd.read_csv(path)
            if 'Timestamp' in df.columns: df = df.drop(columns=['Timestamp'])
            
            X_df = df.iloc[:, :-1]
            y_raw = df.iloc[:, -1]
            y_flat = y_raw.map(self.label_map).fillna(0).values.astype(int)
            
            X_flat = self.scaler.fit_transform(X_df.values)
            num_samples = len(X_flat)
            true_feats = X_flat.shape[1]
            target_feats = 18
            
            if true_feats > target_feats: X_flat = X_flat[:, :target_feats]
            elif true_feats < target_feats:
                pad = np.zeros((num_samples, target_feats - true_feats))
                X_flat = np.hstack((X_flat, pad))

            pad_len = (self.slice_len - (num_samples % self.slice_len)) % self.slice_len
            if pad_len > 0:
                pad_X = np.zeros((pad_len, target_feats))
                pad_y = np.zeros(pad_len, dtype=int)
                X_flat = np.concatenate([X_flat, pad_X], axis=0)
                y_flat = np.concatenate([y_flat, pad_y], axis=0)

            X = torch.tensor(X_flat, dtype=torch.float32).view(-1, self.slice_len, target_feats)
            y = torch.tensor(y_flat, dtype=torch.long).view(-1, self.slice_len)[:, -1]

            # 2. Init Model & Weights
            if self.net is None:
                self.net = ConvNN(slice_len=self.slice_len, num_feats=target_feats, classes=3)
            
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.net.load_state_dict(state_dict, strict=True)

            # 3. Train
            optimizer = optim.Adam(self.net.parameters(), lr=0.001) 
            criterion = nn.NLLLoss() 
            self.net.train()
            for _ in range(3):
                optimizer.zero_grad()
                out = self.net(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            # 4. Save Metrics to SAFE STORAGE
            acc = self.calculate_accuracy(X, y)
            print(f"Round {rnd} | Loss: {loss.item():.4f} | Accuracy: {acc:.2f}%")
            
            self.history.append({"round": rnd, "loss": loss.item(), "accuracy": acc})
            
            # SAVE TO /app/results (The Persistent Volume)
            with open("/app/results/metrics.json", "w") as f:
                json.dump(self.history, f)

            return [val.cpu().numpy() for _, val in self.net.state_dict().items()], len(X), {}
            
        except Exception:
            import traceback
            traceback.print_exc()
            return parameters, 0, {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {"accuracy": 0.0}

if __name__ == "__main__":
    pod_name = os.getenv("POD_NAME", "client-0")
    client_id = pod_name.split("-")[-1]
    
    # Ensure the safe folder exists
    os.makedirs("/app/results", exist_ok=True)
    
    fl.client.start_numpy_client(server_address="fl-server:8080", client=CsvClient(client_id))
    
    # Still sleep to allow manual copying comfortably
    print("Training Finished. Metrics saved to /app/results/metrics.json")
    time.sleep(3600)

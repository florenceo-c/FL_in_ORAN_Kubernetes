import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 1. Load your raw data
# Replace 'combined_slices.csv' with your actual filename if different
df = pd.read_csv('combined_slices.csv')

# 2. Split: 20% for Global Server Test, 80% for Clients
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['slice_type'])

# 3. Save the Global Test Set
os.makedirs("data/server", exist_ok=True)
test_df.to_csv("data/server/test.csv", index=False)
print(f"Global Test Set saved: {len(test_df)} samples")

# 4. Split the 80% among 3 Clients
clients = 3
client_data = np.array_split(train_df, clients)

# 5. Save Client Data
for i in range(clients):
    os.makedirs(f"data/client_{i}", exist_ok=True)
    # Save as round_1.csv (Simulating static data for now)
    # If you want dynamic rounds, we can split this further, but let's keep it simple
    client_data[i].to_csv(f"data/client_{i}/round_1.csv", index=False)
    # Copy same data for other rounds just to keep the file structure valid
    for r in range(2, 6):
        client_data[i].to_csv(f"data/client_{i}/round_{r}.csv", index=False)
    
    print(f"Client {i} received {len(client_data[i])} samples")

print("Data splitting complete.")

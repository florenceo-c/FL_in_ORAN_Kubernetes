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

# 4. Separate classes
mmtc_df = train_df[train_df['slice_type'] == 'mmtc'].sample(frac=1, random_state=42)
embb_df = train_df[train_df['slice_type'] == 'embb'].sample(frac=1, random_state=42)
urllc_df = train_df[train_df['slice_type'] == 'urllc'].sample(frac=1, random_state=42)

# Helper function
def split_class(df_class, proportions):
    n = len(df_class)
    n0 = int(proportions[0] * n)
    n1 = int(proportions[1] * n)
    n2 = n - n0 - n1
    return [
        df_class.iloc[:n0],
        df_class.iloc[n0:n0+n1],
        df_class.iloc[n0+n1:]
    ]

# 5. Create label-skew non-IID partitions
# Client 0 gets most of mmtc
mmtc_parts = split_class(mmtc_df, [0.7, 0.15, 0.15])

# Client 1 gets most of embb
embb_parts = split_class(embb_df, [0.15, 0.7, 0.15])

# Client 2 gets most of urllc
urllc_parts = split_class(urllc_df, [0.15, 0.15, 0.7])

client_0 = pd.concat([mmtc_parts[0], embb_parts[0], urllc_parts[0]]).sample(frac=1, random_state=42)
client_1 = pd.concat([mmtc_parts[1], embb_parts[1], urllc_parts[1]]).sample(frac=1, random_state=42)
client_2 = pd.concat([mmtc_parts[2], embb_parts[2], urllc_parts[2]]).sample(frac=1, random_state=42)

client_data = [client_0, client_1, client_2]

# 5. Save Client Data
for i in range(3):
    os.makedirs(f"data/client_{i}", exist_ok=True)
    client_data[i].to_csv(f"data/client_{i}/round_1.csv", index=False)
    for r in range(2, 6):
        client_data[i].to_csv(f"data/client_{i}/round_{r}.csv", index=False)
    print(f"Client {i} received {len(client_data[i])} samples")
    print(client_data[i]['slice_type'].value_counts())


print("Data splitting complete.")


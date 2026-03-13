import pandas as pd

for i in range(3):
    df = pd.read_csv(f"data/client_{i}/round_1.csv")
    print(f"\nClient {i}")
    print(df['slice_type'].value_counts())

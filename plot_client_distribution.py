import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Client 0": {"mmtc": 8192, "embb": 529, "urllc": 161},
    "Client 1": {"mmtc": 1755, "embb": 2468, "urllc": 161},
    "Client 2": {"mmtc": 1757, "embb": 530, "urllc": 756},
}

df = pd.DataFrame(data).T
df.plot(kind="bar", figsize=(8,5))
plt.title("Non-IID Class Distribution Across Clients")
plt.xlabel("Client")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("client_distribution.png")
print("Saved figure: client_distribution.png")
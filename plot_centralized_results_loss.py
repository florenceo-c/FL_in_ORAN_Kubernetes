import pandas as pd
import matplotlib.pyplot as plt

# Load saved metrics
df = pd.read_csv("centralized_baseline_metrics.csv")

fig, ax = plt.subplots(1, 2, figsize=(12,5))

# -----------------------------
# Accuracy plot
# -----------------------------
ax[0].plot(df["epoch"], df["train_accuracy"], marker="o", label="Train Accuracy")
ax[0].plot(df["epoch"], df["test_accuracy"], marker="s", label="Test Accuracy")

ax[0].set_title("Accuracy vs Epoch")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy (%)")
ax[0].legend()
ax[0].grid(True)

# -----------------------------
# Loss plot
# -----------------------------
ax[1].plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
ax[1].plot(df["epoch"], df["test_loss"], marker="s", label="Test Loss")

ax[1].set_title("Loss vs Epoch")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig("centralized_training_curves.png")

print("Saved figure: centralized_training_curves.png")
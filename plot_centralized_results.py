import pandas as pd
import matplotlib.pyplot as plt

# Load saved metrics
df = pd.read_csv("centralized_baseline_metrics.csv")

# Plot accuracy curves
plt.figure(figsize=(8,5))

plt.plot(df["epoch"], df["train_accuracy"], marker="o", label="Train Accuracy")
plt.plot(df["epoch"], df["test_accuracy"], marker="s", label="Test Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Centralized Baseline Learning Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("centralized_learning_curve.png")

print("Saved figure: centralized_learning_curve.png")
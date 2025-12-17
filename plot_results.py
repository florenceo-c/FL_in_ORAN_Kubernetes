import json
import matplotlib.pyplot as plt
import os

def load_data(filename, client_name):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Skipping {client_name}.")
        return None, None, None

    with open(filename, 'r') as f:
        data = json.load(f)
    return [d['round'] for d in data], [d['accuracy'] for d in data], [d['loss'] for d in data]

# Setup Plot
plt.figure(figsize=(10, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['o', 'x', 's']
filenames = ['metrics_client_0.json', 'metrics_client_1.json', 'metrics_client_2.json']

# Plot all 3 clients
for i in range(3):
    rounds, acc, loss = load_data(filenames[i], f"Client {i}")

    if rounds:
        plt.plot(rounds, acc, marker=markers[i], color=colors[i], label=f'Client {i}')

plt.title('Federated Learning Accuracy (O-RAN)')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('fl_final_results.png')
print("Graph saved as fl_final_results.png")

# Overview

This project implements a federated learning (FL) system for classifying network traffic slices in an O-RAN (Open Radio Access Network) environment. The model distinguishes between three 5G service categories:

- **mMTC** (massive machine-type communications)
  
- **eMBB** (enhanced mobile broadband)

- **URLLC** (ultra-reliable low-latency communications)

The system is built using:

* Flower for federated orchestration

* PyTorch for model training

This repository currently covers the first planned activity of the project:

* Federated learning environment setup

* Non-IID client data partitioning

* Centralized baseline model validation

# Project Structure
``` Plain text
├── client.py                    # Flower client (local training)
├── server.py                    # Flower server (FedAvg aggregation)
├── cnn_model.py                 # CNN architecture
├── splitter.py                  # Data partitioning (non-IID)
├── centralized_baseline.py      # Centralized training script
├── plot_centralized_results.py  # Accuracy/loss curves
├── plot_confusion_matrix.py     # Confusion matrix visualization
├── data/
│   ├── client_0/
│   ├── client_1/
│   ├── client_2/
│   └── server/
├── centralized_baseline_metrics.csv
├── centralized_learning_curve.png
├── centralized_training_curves.png
├── centralized_confusion_matrix.png
└── requirements.txt
```
# Environment Setup

Install dependencies:
``` Bash
pip install -r requirements.txt
```
Core dependencies include:
* flwr
* torch
* pandas
* scikit-learn
* matplotlib
* seaborn

# Federated Learning Setup

The federated learning system consists of:

- **Server** (```server.py```)
  * Implements FedAvg aggregation
  * Evaluates global model on a held-out test set

- **Clients** (```client.py```)
  * Train locally on partitioned datasets
  * Send updated model weights to the server

- **Model** (```cnn_model.py```)
  * Convolutional neural network operating on sliced feature inputs
    
# Non-IID Data Partitioning

The dataset is partitioned across clients using label-skew non-IID splitting.
Each client has a different class distribution:

* Client 0 → dominated by **mMTC**

* Client 1 → dominated by **eMBB**

* Client 2 → relatively enriched in **URLLC**

This setup reflects realistic federated environments where data distributions differ across nodes.

To regenerate the partitions:
``` Bash
python splitter.py
```
## Centralized Baseline (Risk Mitigation)

To verify that the model can learn the dataset independently of federated constraints, a centralized baseline experiment was implemented.
Run:
``` Bash
python centralized_baseline.py
```
This trains the CNN using the combined client datasets and evaluates on the server test set.

   **Results**
* Final test accuracy: ~85%
* Stable convergence observed across epochs
  
# Evaluation and Visualization

 **Learning Curves**
 
Generate accuracy and loss plots:
``` Bash
python plot_centralized_results.py
```
Output:
``` Plain text
centralized_training_curves.png
```
This shows:
 * Training vs test accuracy
* Training vs test loss
  
 **Confusion Matrix**
  
  Generate confusion matrix:
``` Bash
python plot_confusion_matrix.py
```
Output:
``` Plain text
centralized_confusion_matrix.png
```
Key observation:
* Strong performance on **mMTC** and **URLLC**
* Misclassification primarily occurs between **eMBB** and **mMTC**
* Indicates class imbalance and feature overlap effects
# Contributors
 * Veera Tummala
 * Florence Onyike

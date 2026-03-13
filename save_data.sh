#!/bin/bash
echo "Waiting for training to complete..."

# Loop until we see "Training Finished" in the logs
while true; do
    if kubectl logs fl-client-0 | grep -q "Training Finished"; then
        echo "Training Complete! Downloading data..."
        break
    fi
    sleep 5
done

# Wait a few seconds for files to flush to disk
sleep 5

# Copy from the SAFE storage path
echo "Saving Client 0..."
kubectl cp fl-client-0:/app/results/metrics.json ./metrics_client_0.json

echo "Saving Client 1..."
kubectl cp fl-client-1:/app/results/metrics.json ./metrics_client_1.json

echo "Saving Client 2..."
kubectl cp fl-client-2:/app/results/metrics.json ./metrics_client_2.json

echo "All data saved successfully!"

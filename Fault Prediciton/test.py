import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time
from flask import Flask, jsonify
import threading
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
latest_mse = None  # Global variable to store the latest MSE value
latest_prediction = None  # Global variable to store the latest prediction
latest_target = None  # Global variable to store the latest actual target

# Define the LSTM model for live anomaly detection
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, output_size=3 * 5):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Initialize model, criterion, and optimizer
input_length = 60
output_length = 5
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
threshold = 0.25
streaming_data_buffer = deque(maxlen=input_length + output_length)

# Generate synthetic vibrational data
def get_live_vibration_data(time_steps=100, noise_level=0.1, fault=False):
    t = np.linspace(0, 10, time_steps)
    frequency = 0.5
    if fault:
        stretch_factor = 3 * np.linspace(0, 1, time_steps)
        x = np.sin(2 * np.pi * frequency * t * stretch_factor) + noise_level * np.random.randn(time_steps)
        y = np.sin(2 * np.pi * frequency * t * stretch_factor + np.pi / 4) + noise_level * np.random.randn(time_steps)
        z = np.sin(2 * np.pi * frequency * t * stretch_factor + np.pi / 2) + noise_level * np.random.randn(time_steps)
    else:
        x = np.sin(2 * np.pi * frequency * t) + noise_level * np.random.randn(time_steps)
        y = np.sin(2 * np.pi * frequency * t + np.pi / 4) + noise_level * np.random.randn(time_steps)
        z = np.sin(2 * np.pi * frequency * t + np.pi / 2) + noise_level * np.random.randn(time_steps)
    return np.stack([x, y, z], axis=-1)

# Flask route to retrieve the latest MSE, prediction, and target
@app.route('/get_mse', methods=['GET'])
def get_mse():
    return jsonify({
        'mse': latest_mse,
        'prediction': latest_prediction.tolist() if latest_prediction is not None else None,
        'target': latest_target.tolist() if latest_target is not None else None
    })

# Background thread to train on non-fault data initially, then dynamically predict and adapt with fault data
def run_anomaly_detection():
    global latest_mse, latest_prediction, latest_target
    initial_training_data = []

    # Collect initial 100 non-fault samples for model training
    while len(initial_training_data) < 100:
        non_fault_data = get_live_vibration_data(fault=False)
        initial_training_data.extend(non_fault_data.tolist())

    # Train on the first 100 non-fault data
    initial_training_data = torch.tensor(initial_training_data[:100], dtype=torch.float32).reshape(1, 100, 3)
    for epoch in range(10):  # Train for a few epochs for stability
        optimizer.zero_grad()
        prediction = model(initial_training_data[:, :input_length, :])
        target = initial_training_data[:, input_length:input_length + output_length, :].reshape(1, -1)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

    print("Initial training completed.")

    # Start dynamic prediction and adaptation
    while True:
        new_data = get_live_vibration_data(fault=np.random.rand() < 0.1)  # 10% chance of fault
        streaming_data_buffer.extend(new_data)

        if len(streaming_data_buffer) >= input_length + output_length:
            current_window = torch.tensor([list(streaming_data_buffer)[:input_length]], dtype=torch.float32)
            with torch.no_grad():
                prediction = model(current_window)
                target = torch.tensor([list(streaming_data_buffer)[input_length:]], dtype=torch.float32).reshape(1, -1)
                mse = criterion(prediction, target).item()

                # Store the latest prediction and target
                latest_prediction = prediction.numpy()
                latest_target = target.numpy()
                latest_mse = mse

            # Check if the MSE exceeds the threshold
            if latest_mse > threshold:
                # Dynamically adapt by feeding back the fault data for further tuning
                print("Anomaly detected, adapting model with fault data.")
                current_window.requires_grad = True  # Mark as requires grad for further training
                optimizer.zero_grad()
                prediction = model(current_window)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()

                # Plotting current prediction vs. target for quick visualization
                plt.figure(figsize=(10, 4))
                plt.plot(target.numpy().flatten(), label="Target")
                plt.plot(prediction.numpy().flatten(), label="Prediction", linestyle="--")
                plt.legend()
                plt.title(f"Instance with MSE={latest_mse:.4f}")
                plt.show()

            # Remove output_length entries to simulate moving window
            for _ in range(output_length):
                streaming_data_buffer.popleft()
                
        time.sleep(1)

# Start the background anomaly detection thread
anomaly_thread = threading.Thread(target=run_anomaly_detection)
anomaly_thread.start()

# Run Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

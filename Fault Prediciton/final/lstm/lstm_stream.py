# server_mse_stream.py
import numpy as np
import torch
from collections import deque
import time
from flask import Flask, jsonify
import threading
from lstm_load import load_model  # Import the load_model function from lstm_model.py
# lstm_model.py
import torch
import torch.nn as nn

# Initialize Flask app
app = Flask(__name__)
latest_mse = None  # Global variable to store the latest MSE value
latest_prediction = None  # Global variable to store the latest prediction
latest_target = None  # Global variable to store the latest actual target

# Initialize model and load the trained weights
model = load_model('/Users/prasanna/Desktop/Hack2Future/file/lstm_model.pth')  # Load your trained model here
criterion = nn.MSELoss()
threshold = 0.25
input_length = 80
output_length = 20
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

# Background thread to run live anomaly detection
def run_anomaly_detection():
    global latest_mse, latest_prediction, latest_target
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
            print(f"Latest MSE: {latest_mse}")
            print(f"Latest pred: {prediction}")
            print(f"Latest tar: {target}")  # Print for server logs
            for _ in range(output_length):
                streaming_data_buffer.popleft()
        time.sleep(1)

# Start the background anomaly detection thread
anomaly_thread = threading.Thread(target=run_anomaly_detection)
anomaly_thread.start()

# Run Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

# ml_model.py
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# Define the LSTM model for live anomaly detection
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, output_size=3 * 20):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Initialize model, criterion, and optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
threshold = 0.25

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

# Function to perform live anomaly detection
def run_anomaly_detection(streaming_data_buffer, input_length, output_length):
    while True:
        new_data = get_live_vibration_data(fault=np.random.rand() < 0.1)  # 10% chance of fault
        streaming_data_buffer.extend(new_data)
        if len(streaming_data_buffer) >= input_length + output_length:
            current_window = torch.tensor([list(streaming_data_buffer)[:input_length]], dtype=torch.float32)
            with torch.no_grad():
                prediction = model(current_window)
                target = torch.tensor([list(streaming_data_buffer)[input_length:]], dtype=torch.float32).reshape(1, -1)
                mse = criterion(prediction, target).item()

                latest_prediction = prediction.numpy()
                latest_target = target.numpy()
            latest_mse = mse
            for _ in range(output_length):
                streaming_data_buffer.popleft()

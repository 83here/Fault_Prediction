import torch
import numpy as np
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn  # Make sure to import nn module

# Define the LSTM model as before
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, output_size=3 * 20):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Get the output of the last time step
        x = self.fc(x)
        return x

# Initialize model, criterion, and optimizer
input_length = 80
output_length = 20
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to generate non-fault periodic vibrational data
def generate_non_fault_data(time_steps=100, noise_level=0.1):
    t = np.linspace(0, 10, time_steps)
    frequency = 0.5
    x = np.sin(2 * np.pi * frequency * t) + noise_level * np.random.randn(time_steps)
    y = np.sin(2 * np.pi * frequency * t + np.pi / 4) + noise_level * np.random.randn(time_steps)
    z = np.sin(2 * np.pi * frequency * t + np.pi / 2) + noise_level * np.random.randn(time_steps)
    return np.stack([x, y, z], axis=-1)

# Prepare dataset with targets being the same as inputs
def prepare_dataset(num_samples=500):
    inputs = []
    targets = []
    for _ in range(num_samples):
        # Generate non-fault data
        non_fault_data = generate_non_fault_data(input_length + output_length)

        # Split into input and target sequences
        inputs.append(non_fault_data[:input_length])  # Input part
        targets.append(non_fault_data[input_length:])  # Target part (the next part of the same data)

    # Convert lists to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32).reshape(num_samples, -1)  # Flatten targets
    return TensorDataset(inputs, targets)

# Training loop
def train_model(model, criterion, optimizer, num_epochs=50, batch_size=32):
    dataset = prepare_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

# Run training
train_model(model, criterion, optimizer)

# Save the model state dictionary locally
model_path = "/Users/prasanna/Desktop/Hack2Future/final/lstm/lstm_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

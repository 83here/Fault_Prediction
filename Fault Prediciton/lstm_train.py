import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# Define the LSTM model
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

# Load non-fault data from CSV
def load_non_fault_data(file_path, input_length=80, output_length=20):
    data = pd.read_csv(file_path)
    sequences = []
    for i in range(len(data) - (input_length + output_length)):
        sequence = data.iloc[i:i + input_length + output_length][['x', 'y', 'z']].values
        sequences.append(sequence)

    inputs = [seq[:input_length] for seq in sequences]
    targets = [seq[input_length:] for seq in sequences]
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32).reshape(len(sequences), -1)
    return TensorDataset(inputs, targets)

# Training loop
def train_model(model, criterion, optimizer, dataset, num_epochs=50, batch_size=32):
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

# Load the non-fault dataset from CSV and train the model
non_fault_data_file = "/Users/prasanna/Desktop/Hack2Future/non_fault_data.csv"  # Replace with the actual path to your CSV file
non_fault_dataset = load_non_fault_data(non_fault_data_file, input_length, output_length)
train_model(model, criterion, optimizer, non_fault_dataset)

# Save the model state dictionary locally
model_path = "/Users/prasanna/Desktop/Hack2Future/lstm_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

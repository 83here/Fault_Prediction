import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define the dataset class
class VibrationDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.scaler = MinMaxScaler()
        # Scale the data using only the x, y, z columns
        self.data = self.scaler.fit_transform(data[['x', 'y', 'z']].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Define the complex autoencoder model
class ComplexAutoencoder(nn.Module):
    def __init__(self):
        super(ComplexAutoencoder, self).__init__()
        # Encoder with more layers and dropout
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Decoder with more layers and dropout
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load datasets
non_fault_dataset = VibrationDataset('/Users/prasanna/Desktop/Hack2Future/final/non_fault_data.csv')  # Non-fault dataset for training
fault_dataset = VibrationDataset('/Users/prasanna/Desktop/Hack2Future/final/fault_data.csv')  # Fault dataset for evaluation

# Create data loaders
non_fault_loader = DataLoader(non_fault_dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = ComplexAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the autoencoder only on non-fault data
num_epochs = 20
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for data in non_fault_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)  # Reconstruction loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(non_fault_loader):.4f}")

# Evaluate reconstruction error on fault data
fault_loader = DataLoader(fault_dataset, batch_size=32, shuffle=False)
model.eval()
reconstruction_errors = []
with torch.no_grad():
    for data in fault_loader:
        output = model(data)  # Get reconstructed output
        loss = criterion(output, data)  # Calculate reconstruction error
        reconstruction_errors.append(loss.item())

# Print the average reconstruction error on fault data
print(f"Average reconstruction error on fault data: {np.mean(reconstruction_errors):.4f}")

# Save the model state dictionary locally
model_path = "/Users/prasanna/Desktop/Hack2Future/final/encoder/complex_autoencoder_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

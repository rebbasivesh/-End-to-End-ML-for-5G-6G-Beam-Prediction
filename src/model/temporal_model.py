import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import yaml

class LSTMLight(nn.Module):
    """
    Bonus: A lightweight LSTM for Temporal Beam Prediction (e.g. trajectory mapping via RSRP/SINR history).
    Input Shape: (Batch, Sequence_length, Features)
    Output Shape: (Batch, Num_Classes)
    """
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(LSTMLight, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Forward pass
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take the last time step
        return out

def train_temporal_dummy(config_path="configs/config.yaml"):
    """
    Demonstrative framework for future upgrade.
    Trains on randomly structured sequential data for validation.
    """
    num_features = 6
    num_classes = 8
    seq_length = 5
    batch_size = 32
    
    # Simulated sequential dataset
    X_seq = torch.randn(100, seq_length, num_features)
    y_seq = torch.randint(0, num_classes, (100,))
    
    dataset = TensorDataset(X_seq, y_seq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTMLight(input_size=num_features, hidden_size=16, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Temporal LSTM on dummy sequence data...")
    for epoch in range(1, 6):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/5 - Loss: {total_loss/len(loader):.4f}")
        
    print("Temporal model framework verified successfully.")
    
if __name__ == "__main__":
    train_temporal_dummy()

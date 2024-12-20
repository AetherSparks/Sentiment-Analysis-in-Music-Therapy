import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import joblib

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the dataset
print("Loading dataset...")
file_path = "Original/therapeutic_music_enriched.csv"
data = pd.read_csv(file_path)

# Encode categorical target: Mood_Label
print("Encoding target labels...")
label_encoder = LabelEncoder()
data['Mood_Label'] = label_encoder.fit_transform(data['Mood_Label'])

# Define numerical features
numerical_features = [
    'Track Popularity', 'Danceability', 'Energy', 'Key', 'Loudness', 
    'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 
    'Valence', 'Tempo', 'Duration (ms)'
]

# Scale numerical features
print("Scaling numerical features...")
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Features and target
X = data[numerical_features].values
y = data['Mood_Label'].values

# Train-test split
print("Splitting data into train and test sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define custom dataset
class MusicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create DataLoader
train_dataset = MusicDataset(X_train, y_train)
val_dataset = MusicDataset(X_val, y_val)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define CNN model for numerical data
class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_dim * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        return self.network(x)

# Initialize model, optimizer, and loss function
input_dim = X_train.shape[1]
num_classes = len(np.unique(y))
model = CNNModel(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
patience_counter = 0

# Training loop
print("Starting training...")
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_accuracy = 100. * correct / total
    print(f"Epoch {epoch+1}: Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Validation step
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    val_accuracy = 100. * val_correct / val_total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print("Validation loss improved. Saving model...")
        torch.save(model.state_dict(), "CNN/best_cnn_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Dynamically get the unique classes in y_true and y_pred
unique_labels = np.unique(np.concatenate([y_true, y_pred]))

# Generate classification report using only the relevant labels
print("\nFinal Evaluation on Validation Set:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, labels=unique_labels))

# Save preprocessing objects
print("Saving preprocessing objects...")
joblib.dump(scaler, "CNN/cnn_scaler.pkl")
joblib.dump(label_encoder, "CNN/cnn_label_encoder.pkl")

print("Training and evaluation complete.")

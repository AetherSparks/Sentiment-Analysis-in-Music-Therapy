import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import joblib
import time

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
print("Loading dataset...")
file_path = "Original/therapeutic_music_enriched.csv"
data = pd.read_csv(file_path)

# Encode target labels
print("Encoding target labels...")
label_encoder = LabelEncoder()
data['Mood_Label'] = label_encoder.fit_transform(data['Mood_Label'])

# Define text and numerical features
text_features = ['Track Name', 'Artist Name', 'Artist Genres', 'Album']
numerical_features = [
    'Track Popularity', 'Danceability', 'Energy', 'Key', 'Loudness', 
    'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 
    'Valence', 'Tempo', 'Duration (ms)'
]

# Scale numerical features
print("Scaling numerical features...")
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Combine text features into a single string
data['Combined_Text'] = data[text_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Train-test split
X_text = data['Combined_Text'].values
X_num = data[numerical_features].values
y = data['Mood_Label'].values

X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Custom Dataset
class MusicDataset(Dataset):
    def __init__(self, texts, numerical, labels, tokenizer, max_length=128):
        self.texts = texts
        self.numerical = numerical
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        numerical = self.numerical[idx]
        label = self.labels[idx]

        # Tokenize text
        encoded = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "numerical_features": torch.tensor(numerical, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Create DataLoader
train_dataset = MusicDataset(X_text_train, X_num_train, y_train, tokenizer)
val_dataset = MusicDataset(X_text_val, X_num_val, y_val, tokenizer)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define BERT + Numerical Model
class BERTWithNumerical(nn.Module):
    def __init__(self, num_numerical_features, num_classes):
        super(BERTWithNumerical, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc_numerical = nn.Linear(num_numerical_features, 128)
        self.classifier = nn.Linear(768 + 128, num_classes)  # Combine BERT and numerical outputs

    def forward(self, input_ids, attention_mask, numerical_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_output.pooler_output  # [CLS] token output
        numerical_out = torch.relu(self.fc_numerical(numerical_features))
        combined = torch.cat((bert_cls, numerical_out), dim=1)
        return self.classifier(combined)

# Initialize model, optimizer, and loss function
num_classes = len(np.unique(y))
model = BERTWithNumerical(num_numerical_features=len(numerical_features), num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop with early stopping
patience = 5
best_val_loss = float('inf')
patience_counter = 0

print("Starting training...")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        numerical_features = batch["numerical_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, numerical_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_accuracy = 100. * correct / total
    print(f"Epoch {epoch+1}: Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical_features = batch["numerical_features"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, numerical_features)
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
        torch.save(model.state_dict(), "BERT/best_bert_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Final evaluation
print("\nFinal Evaluation on Validation Set:")
unique_labels = np.unique(np.concatenate([y_true, y_pred]))
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, labels=unique_labels, zero_division=0))

# Save preprocessing objects
joblib.dump(scaler, "BERT/bert_scaler.pkl")
joblib.dump(label_encoder, "BERT/bert_label_encoder.pkl")

print("Training and evaluation complete.")

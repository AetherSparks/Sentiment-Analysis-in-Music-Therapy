import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import time
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Measure time for the entire process
start_time = time.time()

# Directory to save the best model
MODEL_SAVE_PATH = "./DEBERTA/deberta_model.pt"


def main():
    # Load and preprocess the dataset
    print("Loading dataset...")
    file_path = "Original/therapeutic_music_enriched.csv"
    data = pd.read_csv(file_path)

    # Encode categorical target: Mood_Label
    print("Encoding target labels...")
    label_encoder = LabelEncoder()
    data['Mood_Label'] = label_encoder.fit_transform(data['Mood_Label'])

    # Separate text and numerical features
    text_features = ['Track Name', 'Artist Name', 'Artist Genres', 'Album']
    numerical_features = [
        'Track Popularity', 'Danceability', 'Energy', 'Key', 'Loudness', 
        'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 
        'Valence', 'Tempo', 'Duration (ms)'
    ]

    # Preprocess numerical features
    print("Scaling numerical features...")
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Define custom dataset
    class MusicDataset(Dataset):
        def __init__(self, text_data, numerical_data, labels, tokenizer, max_length=128):
            self.text_data = text_data
            self.numerical_data = numerical_data
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.text_data)

        def __getitem__(self, idx):
            text = self.text_data[idx]
            numerical = self.numerical_data[idx]
            label = self.labels[idx]

            # Tokenize text
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "numerical_features": torch.tensor(numerical, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long),
            }

    # Load DeBERTa tokenizer and model
    print("Loading DeBERTa model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir="./DEBERTA/deberta_cache")
    text_model = AutoModel.from_pretrained("microsoft/deberta-base", cache_dir="./DEBERTA/deberta_cache").to(device)

    # Combine text features into a single column
    text_data = data[text_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
    numerical_data = data[numerical_features].values
    labels = data['Mood_Label'].values

    # Train-test split
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        text_data, numerical_data, labels, test_size=0.2, random_state=42
    )

    # Create PyTorch datasets and dataloaders
    batch_size = 32
    train_dataset = MusicDataset(X_text_train, X_num_train, y_train, tokenizer)
    test_dataset = MusicDataset(X_text_test, X_num_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Neural network with Dropout
    class CombinedModel(torch.nn.Module):
        def __init__(self, text_model, numerical_input_dim, output_dim):
            super(CombinedModel, self).__init__()
            self.text_model = text_model
            self.fc_text = torch.nn.Linear(text_model.config.hidden_size, 128)
            self.dropout = torch.nn.Dropout(0.5)
            self.fc_num = torch.nn.Linear(numerical_input_dim, 64)
            self.fc_combined = torch.nn.Linear(128 + 64, output_dim)

        def forward(self, input_ids, attention_mask, numerical_features):
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embedding = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
            text_features = torch.relu(self.fc_text(text_embedding))
            text_features = self.dropout(text_features)

            numerical_features = torch.relu(self.fc_num(numerical_features))
            combined = torch.cat((text_features, numerical_features), dim=1)
            output = self.fc_combined(combined)
            return output

    model = CombinedModel(text_model, numerical_data.shape[1], len(label_encoder.classes_)).to(device)

    # Define optimizer and loss function with L2 regularization
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)

    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    print("Starting training...")
    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numerical_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Accuracy={accuracy:.2f}%")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                numerical_features = batch['numerical_features'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, numerical_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100. * val_correct / val_total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    print("Training completed!")


if __name__ == "__main__":
    main()

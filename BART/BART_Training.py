import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForSequenceClassification
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import time

print(torch.cuda.is_available())

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BART tokenizer and model (use a smaller variant if needed)
print("Loading BART model and tokenizer...")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", cache_dir="./BART/bart_cache")
model = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=12).to(device)

# Collate function to handle batching
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    numerical_features = torch.stack([item['numerical_features'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'numerical_features': numerical_features,
        'labels': labels
    }

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
                padding='max_length',
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "numerical_features": torch.tensor(numerical, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long),
            }

    # Combine text features into a single column
    text_data = data[text_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
    numerical_data = data[numerical_features].values
    labels = data['Mood_Label'].values

    # Train-test split
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        text_data, numerical_data, labels, test_size=0.2, random_state=42
    )

    # Create PyTorch datasets
    train_dataset = MusicDataset(X_text_train, X_num_train, y_train, tokenizer)
    test_dataset = MusicDataset(X_text_test, X_num_test, y_test, tokenizer)

    # Create DataLoader with collate function
    batch_size = 64  # Increased batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Define optimizer and loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    print("Starting training...")
    num_epochs = 10  # Reduced epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        start_epoch = time.time()

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.logits.argmax(dim=1) == labels).sum().item()

        # Calculate training accuracy
        accuracy = 100. * correct / total
        epoch_time = time.time() - start_epoch
        print(f"Training Epoch {epoch + 1} finished: Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")

        # Validation step
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validating", unit="batch"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                numerical_features = batch['numerical_features'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

                # Store predictions and true labels for metrics
                y_pred.extend(outputs.logits.argmax(dim=1).cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        # Calculate validation accuracy
        val_accuracy = 100. * val_correct / val_total
        print(f"Validation: Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Print classification report
        # Print classification report
        unique_labels = np.unique(y_true)
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, labels=unique_labels))


        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print("Validation loss improved, saving model...")
            model.save_pretrained("best_model")
            tokenizer.save_pretrained("best_model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, labels=unique_labels))

    print("Training complete.")

if __name__ == "__main__":
    main()

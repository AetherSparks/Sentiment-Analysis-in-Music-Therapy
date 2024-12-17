import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import time
import joblib

def main():
    # Check for GPU availability
    print("Checking GPU availability...")
    device_type = "cuda" if "gpu" in lgb.__dict__ else "cpu"
    print(f"Using device: {device_type}")

    # Load and preprocess the dataset
    print("Loading dataset...")
    file_path = "Original/therapeutic_music_enriched.csv"
    data = pd.read_csv(file_path)

    # Encode categorical target: Mood_Label
    print("Encoding target labels...")
    label_encoder = LabelEncoder()
    data['Mood_Label'] = label_encoder.fit_transform(data['Mood_Label'])

    # Separate features
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

    # Combine text features into a single column (optional for future use)
    data['Combined_Text'] = data[text_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Features and target
    X = data[numerical_features]  # Use numerical features only
    y = data['Mood_Label']

    # Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, weight=None, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, weight=None, free_raw_data=False, reference=train_data)

    # Define LightGBM parameters
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': len(np.unique(y)),  # Automatically set based on target labels
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'max_depth': 7,
        'verbose': -1  # Suppress LightGBM internal logs
    }

    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    # Training Loop
    print("Starting LightGBM training...")
    num_epochs = 1000  # Max boosting rounds
    start_time = time.time()

    model = None
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training LightGBM", unit="round"):
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=1,
            init_model=model,
            keep_training_booster=True
        )

        # Calculate validation loss
        val_loss = model.best_score['valid_1']['multi_logloss']

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"Validation loss improved at epoch {epoch}. Saving model...")
            model.save_model("LGBM/best_lgbm_model.txt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    end_time = time.time()
    print(f"Training complete in {end_time - start_time:.2f} seconds.")


    # Predict on validation set
    print("Evaluating the model...")
    y_pred = np.argmax(model.predict(X_val), axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')

    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # Get unique labels from LabelEncoder and y_true
    unique_labels = np.unique(np.concatenate([y_val, y_pred]))  # Combine to ensure no missing classes

    # Generate the classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_, labels=unique_labels))


    # Save the model, scaler, and encoder
    print("Saving model and preprocessing objects...")
    joblib.dump(scaler, "LGBM/lgbm_scaler.pkl")
    joblib.dump(label_encoder, "LGBM/lgbm_label_encoder.pkl")

    print("Model training and evaluation complete.")

if __name__ == "__main__":
    main()

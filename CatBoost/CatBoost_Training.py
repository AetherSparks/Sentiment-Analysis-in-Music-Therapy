import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

def main():
    # Load dataset
    print("Loading dataset...")
    file_path = "Original/therapeutic_music_enriched.csv"
    data = pd.read_csv(file_path)

    # Encode target labels
    print("Encoding target labels...")
    label_encoder = LabelEncoder()
    data['Mood_Label'] = label_encoder.fit_transform(data['Mood_Label'])

    # Separate features
    features = [
        'Track Popularity', 'Danceability', 'Energy', 'Key', 'Loudness', 
        'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 
        'Valence', 'Tempo', 'Duration (ms)'
    ]
    X = data[features]
    y = data['Mood_Label']

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train CatBoost classifier
    print("Training CatBoost...")
    model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.1, verbose=0)
    model.fit(X_train, y_train)

    model.save_model("CatBoost/catboost_model.json")
    print("Model saved as catboost_model.json")

    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    all_classes = list(range(len(label_encoder.classes_)))
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_, 
        labels=all_classes, 
        zero_division=0
    ))

if __name__ == "__main__":
    main()
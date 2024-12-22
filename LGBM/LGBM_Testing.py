import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

def test_lgbm_model():
    # Load and preprocess the dataset for testing
    print("Loading dataset for testing...")
    file_path = "Original/therapeutic_music_enriched.csv"
    data = pd.read_csv(file_path)

    # Load saved model, scaler, and label encoder
    print("Loading saved model, scaler, and label encoder...")
    model = lgb.Booster(model_file="LGBM/best_lgbm_model.txt")
    scaler = joblib.load("LGBM/lgbm_scaler.pkl")
    label_encoder = joblib.load("LGBM/lgbm_label_encoder.pkl")

    # Separate features and target
    text_features = ['Track Name', 'Artist Name', 'Artist Genres', 'Album']
    numerical_features = [
        'Track Popularity', 'Danceability', 'Energy', 'Key', 'Loudness', 
        'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 
        'Valence', 'Tempo', 'Duration (ms)'
    ]

    # Preprocess numerical features
    print("Scaling numerical features...")
    data[numerical_features] = scaler.transform(data[numerical_features])

    # Features and target
    X = data[numerical_features]  # Use numerical features only
    y = data['Mood_Label']

    # Encode target labels using the loaded label encoder
    y_encoded = label_encoder.transform(y)

    # Predict using the trained LightGBM model
    print("Making predictions on the test set...")
    y_pred = np.argmax(model.predict(X), axis=1)  # Get the predicted class labels

    # Calculate metrics
    accuracy = accuracy_score(y_encoded, y_pred)
    precision = precision_score(y_encoded, y_pred, average='weighted')
    recall = recall_score(y_encoded, y_pred, average='weighted')
    f1 = f1_score(y_encoded, y_pred, average='weighted')

    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Generate the classification report
    print("\nClassification Report:")
    print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_))

    # Generate and display the confusion matrix
    cm = confusion_matrix(y_encoded, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    cm_display.plot(cmap='Blues', values_format='d')

    print("\nConfusion Matrix:")
    print(cm)

    # Generate the confusion matrix


    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Save the plot as an image file
    plt.savefig('Results/Confusion_Matrices/LGBM_confusion_matrix.png')  # You can change the filename and extension
    plt.close()  # Close the plot to avoid display in the notebook


    print("Testing complete.")

if __name__ == "__main__":
    test_lgbm_model()

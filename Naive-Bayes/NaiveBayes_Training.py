import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

def main():
    # Load dataset
    print("Loading dataset...")
    file_path = "Original/therapeutic_music_enriched.csv"
    data = pd.read_csv(file_path)

    # Encode target labels
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

    # Combine text features into a single column
    data['Combined_Text'] = data[text_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Preprocess numerical features using MinMaxScaler
    print("Scaling numerical features...")
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Split into features and target
    X_text = data['Combined_Text']
    X_num = data[numerical_features]
    y = data['Mood_Label']

    # Split into train and test sets
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize text data using TF-IDF
    print("Vectorizing text data...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_text_train_tfidf = tfidf.fit_transform(X_text_train)
    X_text_test_tfidf = tfidf.transform(X_text_test)

    # Combine numerical and text features
    X_train_combined = hstack([X_text_train_tfidf, X_num_train])
    X_test_combined = hstack([X_text_test_tfidf, X_num_test])

    # Train Naive Bayes Classifier
    print("Training Naive Bayes Classifier...")
    model = MultinomialNB()
    model.fit(X_train_combined, y_train)

    # Make predictions
    print("Evaluating the model...")
    y_pred = model.predict(X_test_combined)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Debugging: Check labels in predictions and true labels
    print(f"Unique classes in y_test: {np.unique(y_test)}")
    print(f"Unique classes in y_pred: {np.unique(y_pred)}")

    # Classification report
    try:
        unique_labels = np.unique(np.concatenate((y_test, y_pred)))
        print("Classification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_, 
            labels=unique_labels, 
            zero_division=0
        ))
    except ValueError as e:
        print(f"Error generating classification report: {e}")
        all_classes = list(range(len(label_encoder.classes_)))
        print("Generating classification report with all classes...")
        print(classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_, 
            labels=all_classes, 
            zero_division=0
        ))

if __name__ == "__main__":
    main()

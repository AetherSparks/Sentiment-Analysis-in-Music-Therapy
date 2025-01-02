import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
print("Loading dataset...")
file_path = "Original/therapeutic_music_enriched.csv"
data = pd.read_csv(file_path)

# Encode target labels
print("Encoding target labels...")
label_encoder = LabelEncoder()
data['Mood_Label'] = label_encoder.fit_transform(data['Mood_Label'])

# Define features and target
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
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load models
print("Loading trained models...")
xgb_model = XGBClassifier()
xgb_model.load_model("XGBoost/xgboost_model.json")

lgb_model = lgb.Booster(model_file="LGBM/best_lgbm_model.txt")

catboost_model = CatBoostClassifier()
catboost_model.load_model("CatBoost/catboost_model.json")

rf_model = joblib.load("Random-Forest/random-forest_model.joblib")

# Define CNN model architecture (match training script)
class CNNModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim * 32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        return self.network(x)

# Recreate CNN model and load weights
print("Recreating CNN model and loading weights...")
input_dim = X_test.shape[1]  # Number of features
num_classes = len(label_encoder.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNNModel(input_dim, num_classes).to(device)
cnn_model.load_state_dict(torch.load("CNN/best_cnn_model.pth"))
cnn_model.eval()

# Define a function for CNN predictions
def predict_cnn(model, X_test):
    model_device = next(model.parameters()).device  # Get the device of the model
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(model_device)  # Move input tensor to model's device
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)
    return predictions.cpu().numpy()  # Return predictions as a CPU numpy array

# Generate predictions
print("Generating predictions...")
xgb_preds = xgb_model.predict(X_test)
lgb_preds = np.argmax(lgb_model.predict(X_test), axis=1)
catboost_preds = catboost_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
cnn_preds = predict_cnn(cnn_model, X_test)

# Debugging: Check shapes of all predictions
print(f"xgb_preds shape: {xgb_preds.shape}")
print(f"lgb_preds shape: {lgb_preds.shape}")
print(f"catboost_preds shape: {catboost_preds.shape}")
print(f"rf_preds shape: {rf_preds.shape}")
print(f"cnn_preds shape: {cnn_preds.shape}")

# Ensure consistent shapes across all predictions
num_samples = len(X_test)
xgb_preds = xgb_preds[:num_samples]
lgb_preds = lgb_preds[:num_samples]
catboost_preds = np.squeeze(catboost_preds)  # Ensure catboost_preds is 1D
catboost_preds = catboost_preds[:num_samples]  # Ensure the correct length
rf_preds = rf_preds[:num_samples]
cnn_preds = cnn_preds[:num_samples]

# Combine predictions using majority voting
print("Combining predictions with majority voting...")
all_preds = np.stack([xgb_preds, lgb_preds, catboost_preds, rf_preds, cnn_preds], axis=1)
final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=all_preds)

# Get unique classes in the test set
unique_classes_in_test = np.unique(y_test)

# Ensure target_names has the correct number of classes
print(f"Number of classes in y_test: {len(unique_classes_in_test)}")
print(f"Number of classes in label_encoder: {len(label_encoder.classes_)}")

# Evaluate the ensemble
print("Evaluating ensemble model...")
accuracy = accuracy_score(y_test, final_preds)
print(f"Hard Voting Ensemble Accuracy: {accuracy:.4f}")
print("\nClassification Report for Hard Voting Ensemble:")

# Use the unique classes from the test set to ensure the correct labels are passed to classification_report
print(classification_report(y_test, final_preds, target_names=label_encoder.classes_[unique_classes_in_test], zero_division=0))

def predict_cnn_probs(model, X_test):
    """
    Generate probability predictions using the CNN model.

    Args:
        model (torch.nn.Module): Trained CNN model.
        X_test (np.ndarray): Test dataset features.

    Returns:
        np.ndarray: Predicted probabilities for each class.
    """
    model_device = next(model.parameters()).device  # Get the device of the model
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(model_device)  # Convert input to a tensor and move to the model's device
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(X_tensor)  # Forward pass
        probs = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities
    return probs.cpu().numpy()  # Convert to NumPy array and return



xgb_probs = xgb_model.predict_proba(X_test)
lgb_probs = lgb_model.predict(X_test)  # LightGBM outputs probabilities
catboost_probs = catboost_model.predict_proba(X_test)
rf_probs = rf_model.predict_proba(X_test)
cnn_probs = predict_cnn_probs(cnn_model, X_test)  # Define a function for CNN probabilities

ensemble_probs = (xgb_probs + lgb_probs + catboost_probs + rf_probs + cnn_probs) / 5
final_preds = np.argmax(ensemble_probs, axis=1)


# Compute Accuracy
accuracy = accuracy_score(y_test, final_preds)
print(f"Soft Voting Ensemble Accuracy: {accuracy:.4f}")

# Compute Classification Report
print("\nClassification Report for Soft Voting Ensemble:")
print(classification_report(y_test, final_preds, target_names=label_encoder.classes_[unique_classes_in_test], zero_division=0))

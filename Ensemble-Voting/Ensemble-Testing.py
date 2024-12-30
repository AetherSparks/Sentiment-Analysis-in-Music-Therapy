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
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



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
input_dim = X.shape[1]  # Number of features
num_classes = len(label_encoder.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNNModel(input_dim, num_classes).to(device)
cnn_model.load_state_dict(torch.load("CNN/best_cnn_model.pth"))
cnn_model.eval()

# Define a function for CNN predictions
def predict_cnn(model, X_data):
    model_device = next(model.parameters()).device  # Get the device of the model
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(model_device)  # Move input tensor to model's device
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)
    return predictions.cpu().numpy()  # Return predictions as a CPU numpy array

# Generate predictions
print("Generating predictions...")
xgb_preds = xgb_model.predict(X)
lgb_preds = np.argmax(lgb_model.predict(X), axis=1)
catboost_preds = catboost_model.predict(X)
rf_preds = rf_model.predict(X)
cnn_preds = predict_cnn(cnn_model, X)
# Squeeze the extra dimension in catboost_preds
catboost_preds = np.squeeze(catboost_preds)

# Debugging: Print shapes of all predictions after adjustments
print(f"xgb_preds shape: {xgb_preds.shape}")
print(f"lgb_preds shape: {lgb_preds.shape}")
print(f"catboost_preds shape: {catboost_preds.shape}")
print(f"rf_preds shape: {rf_preds.shape}")
print(f"cnn_preds shape: {cnn_preds.shape}")

# Combine predictions using majority voting
print("Combining predictions with majority voting...")
all_preds = np.stack([xgb_preds, lgb_preds, catboost_preds, rf_preds, cnn_preds], axis=1)
final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=all_preds)

# Get unique classes in the test set
unique_classes_in_test = np.unique(y)

# Ensure target_names has the correct number of classes
print(f"Number of classes in y_test: {len(unique_classes_in_test)}")
print(f"Number of classes in label_encoder: {len(label_encoder.classes_)}")

# Evaluate the ensemble
print("Evaluating ensemble model...")
accuracy = accuracy_score(y, final_preds)
print(f"Ensemble Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y, final_preds, target_names=label_encoder.classes_[unique_classes_in_test], zero_division=0))



# Generate and display confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y, final_preds)
print("Confusion Matrix:")
print(cm)

# Save the confusion matrix as an image
print("Saving confusion matrix...")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.title("Confusion Matrix - Ensemble Model")
plt.savefig("Results\Confusion_Matrices\Ensemble-Voting_confusion_matrix.png")
plt.show()

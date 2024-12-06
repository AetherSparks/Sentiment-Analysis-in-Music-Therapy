import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
import joblib

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where LGBM_Training.py is located
input_file = os.path.join(base_dir, "../Original/hindi_songs_with_audio.csv")  # Input file path
output_folder = base_dir  # Use the same directory as LGBM_Training.py for outputs

# Load dataset
df = pd.read_csv(input_file)

# Select features
numeric_features = [
    'Artist Popularity', 'Danceability', 'Energy', 'Loudness', 
    'Speechiness', 'Acousticness', 'Instrumentalness', 
    'Liveness', 'Valence', 'Tempo', 'Duration (ms)'
]
categorical_features = ['Key', 'Mode', 'Artist Genres']  # Modify based on your dataset

# Target variable
target = 'Track Popularity'  # Change to your target variable

# Preprocessing: Handle missing values
# Fill numeric columns with their median
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Fill categorical columns with "Unknown"
df[categorical_features] = df[categorical_features].fillna("Unknown")

# Save the processed dataset
processed_csv_path = os.path.join(output_folder, "processed_hindi_songs.csv")
df.to_csv(processed_csv_path, index=False)

# Train-test split
X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Normalize numeric and encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# LightGBM parameters
params = {
    'objective': 'regression',  # Use 'multiclass' if classification
    'metric': 'rmse',  # Use 'multi_logloss' for classification
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Trees
    'num_leaves': 31,  # Controls the complexity of the tree
    'learning_rate': 0.05,
    'feature_fraction': 0.9,  # Controls overfitting
    'early_stopping_round': 50  # Early stopping rounds for the validation set
}

# Train model
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=1000,
)

# Predict on validation set
y_pred = model.predict(X_val)

# Calculate RMSE
rmse = root_mean_squared_error(y_val, y_pred)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save model in binary format
model_path = os.path.join(output_folder, "LGBM_Model.bin")
model.save_model(model_path)

# Save the preprocessor for reuse
preprocessor_path = os.path.join(output_folder, "preprocessor.pkl")
joblib.dump(preprocessor, preprocessor_path)

print(f"Processed CSV saved to: {processed_csv_path}")
print(f"Model saved to: {model_path}")
print(f"Preprocessor saved to: {preprocessor_path}")

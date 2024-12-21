# Sentiment Analysis in Music Therapy

This project focuses on using machine learning models to predict user moods based on audio and metadata from songs. 
By leveraging features like track popularity, danceability, energy, and more, the project aims to classify songs into mood categories for use in music therapy.

## **Project Structure**

### **Directories**

- `Original/`
  - Contains the raw dataset (`therapeutic_music_enriched.csv`) along with other versions of the dataset, including Y.M.I.R(Yielding Melodies for Internal Restoration) Dataset.

- `XGBoost/`
  - Contains the XGBoost model and associated files (e.g., `xgboost_model.json`).

- `LGBM/`
  - Contains the LightGBM model and associated files (e.g., `best_lgbm_model.txt`).

- `CatBoost/`
  - Contains the CatBoost model and associated files (e.g., `catboost_model.json`).

- `Random-Forest/`
  - Contains the Random Forest model and associated files (e.g., `random_forest_model.joblib`).

- `CNN/`
  - Contains the trained CNN model (`best_cnn_model.pth`) and preprocessing files (`cnn_scaler.pkl`, `cnn_label_encoder.pkl`).

- `BERT/`
  - Contains the BERT model training script and files.

- `BART/`
  - Contains the BART model training script and files.

- `DEBERTA/`
  - Contains the DeBERTa model training script and files.

- `ELECTRA/`
  - Contains the ELECTRA model training script and files.

- `INDIC-BERT/`
  - Contains the Indic-BERT model training script and files.

- `Naive-Bayes/`
  - Contains the Naive Bayes classifier training script and files.

- `Ensemble-Voting/`
  - Contains the script for ensemble learning using majority voting.

### **Scripts**
As it is not possible to save the Trained Models in the Repository and upload it to Github due to their large sizes, we have instead uploaded their Training Scripts. Just run the given Files to obtain the trained models along with their corresponding accuracy, f1 score, etc.

1. `XGBoost/XGBoost_Training.py`
   - Script to train and save the XGBoost model.

2. `LGBM/LGBM_Training.py`
   - Script to train and save the LightGBM model.

3. `CatBoost/CatBoost_Training.py`
   - Script to train and save the CatBoost model.

4. `Random-Forest/Random-Forest_Training.py`
   - Script to train and save the Random Forest model.

5. `CNN/CNN_Training.py`
   - Script to train and save the CNN model.

6. `BERT/BERT_Training.py`
   - Script to train and save the BERT model.

7. `BART/BART_Training.py`
   - Script to train and save the BART model.

8. `DEBERTA/DEBERTA_Training.py`
   - Script to train and save the DeBERTa model.

9. `ELECTRA/ELECTRA_Training.py`
   - Script to train and save the ELECTRA model.

10. `INDIC-BERT/INDIC-BERT_Training.py`
    - Script to train and save the Indic-BERT model.

11. `Naive-Bayes/NaiveBayes_Training.py`
    - Script to train and save the Naive Bayes classifier.

12. `Ensemble-Voting/Ensemble-Voting.py`
    - Script to combine predictions from XGBoost, LightGBM, CatBoost, Random Forest, and CNN using majority voting. To run this you must first train the given 5 models.

## **Installation**

### **1. Clone the Repository**
```bash
$ git clone https://github.com/AetherSparks/Sentiment-Analysis-in-Music-Therapy.git
$ cd Sentiment-Analysis-in-Music-Therapy
```

### **2. Create a Virtual Environment**
```bash
$ python -m venv musicvenv
$ source musicvenv/bin/activate  # For Linux/Mac
$ musicvenv\Scripts\activate   # For Windows
```

### **3. Install Dependencies**

Ensure all required libraries are installed by using the `requirements.txt` file:
```bash
$ pip install -r requirements.txt
```

## **Usage**

### **1. Train Individual Models**
Train each model by running their respective training scripts. Example:
```bash
$ python XGBoost/XGBoost_Training.py
$ python LGBM/LGBM_Training.py
$ python CatBoost/CatBoost_Training.py
$ python Random-Forest/RandomForest_Training.py
$ python CNN/CNN_Training.py
$ python BERT/BERT_Training.py
$ python BART/BART_Training.py
$ python DEBERTA/DEBERTA_Training.py
$ python ELECTRA/ELECTRA_Training.py
$ python INDIC-BERT/IndicBERT_Training.py
$ python Naive-Bayes/NaiveBayes_Training.py
```

### **2. Run Ensemble Voting**
Combine predictions using the ensemble script:
```bash
$ python Ensemble-Voting/Ensemble-Voting.py
```

## **Models Used**

1. **XGBoost**
   - Gradient Boosting framework optimized for performance and efficiency.

2. **LightGBM**
   - Fast, distributed gradient boosting framework.

3. **CatBoost**
   - Gradient boosting on decision trees with categorical feature support.

4. **Random Forest**
   - Ensemble learning method using decision trees.

5. **Convolutional Neural Network (CNN)**
   - Deep learning model for analyzing numerical features.

6. **BERT**
   - Transformer model for natural language processing tasks.

7. **BART**
   - Denoising autoencoder for sequence-to-sequence tasks.

8. **DeBERTa**
   - Enhanced BERT model for improved representation.

9. **ELECTRA**
   - Transformer model pre-trained as a discriminator.

10. **Indic-BERT**
    - BERT model optimized for Indian languages.

11. **Naive Bayes**
    - Probabilistic classifier based on Bayes' theorem.

12. **Ensemble Majority Voting**
    - Combines predictions from XGBoost, LightGBM, CatBoost, Random Forest, and CNN to improve accuracy.

## **Dataset**

- **Path**: `Original/therapeutic_music_enriched.csv`
- **Features**:
  - `Track Popularity`, `Danceability`, `Energy`, `Key`, `Loudness`, `Mode`, `Speechiness`, `Acousticness`, `Instrumentalness`, `Liveness`, `Valence`, `Tempo`, `Duration (ms)`
- **Target**: `Mood_Label`

## **Results**

The following accuracies were achieved during testing:

| Model            | Accuracy |
|------------------|----------|
| XGBoost          | 0.97     |
| LightGBM         | 0.97     |
| CatBoost         | 0.96     |
| Random Forest    | 0.92     |
| CNN              | 0.80     |
| BERT             | 0.44     |
| BART             | 0.42     |
| DeBERTa          | 0.35     |
| ELECTRA          | 0.45     |
| Indic-BERT       | 0.46     |
| Naive Bayes      | 0.40     |
| **Ensemble**     | **0.965**|

An In Depth Score of these models have been saved in the `Results` Directory.



### Drive Link for the Datasets and Raw Audio Files
https://drive.google.com/drive/folders/1Iia5wi49W-TZfKnQyKRpj2g0CzBd1yn3?usp=sharing

import sys
import os
import random
import numpy as np
import yaml
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.initializers import GlorotUniform  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE #type: ignore
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Check project structure and import path
try:
    current_directory = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_directory, '../../'))
    if not os.path.basename(project_root) == 'Assignment-1':
        raise FileNotFoundError(
            'Project root "Assignment-1" not found. Ensure the file structure matches the GitHub repository.')
except Exception as e:
    raise RuntimeError(f"Error determining project structure: {e}")

# Load configuration file
yaml_path = os.path.join(project_root, 'dataset_1', 'config.yaml')
try:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file not found at {yaml_path}. Please ensure the file exists.")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing YAML configuration file: {e}")

# Load and preprocess data
def load_and_preprocess_data(file_path):
    """This function takes the file path and performs a series of preprocessing only for dataset 1.

    Returns train and test sets.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {file_path}. Please ensure the file path is correct.")
    except pd.errors.EmptyDataError:
        raise ValueError("Data file is empty.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing CSV file.")

    try:
        # Label encode binary categorical variables
        l = LabelEncoder()
        df['gender'] = l.fit_transform(df['gender'])
        df['ever_married'] = l.fit_transform(df['ever_married'])
        df['Residence_type'] = l.fit_transform(df['Residence_type'])

        # One-hot encode categorical features with more than two categories
        df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)
        df['bmi'].fillna(df['bmi'].median(), inplace=True)

        # Separate features and target variable (stroke)
        X = df.drop(columns=['stroke', 'id'])
        y = df['stroke']

        # Apply Borderline-SMOTE
        b = BorderlineSMOTE(random_state=42)
        X_resampled, y_resampled = b.fit_resample(X, y)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
        )

        # Standardize features
        s = StandardScaler()
        X_train_scaled = s.fit_transform(X_train)
        X_test_scaled = s.transform(X_test)
    except KeyError as e:
        raise KeyError(f"Column missing from dataset during preprocessing: {e}")
    except ValueError as e:
        raise ValueError(f"Error in data processing: {e}")

    return X_train_scaled, X_test_scaled, y_train, y_test

# Set data file path and load data
data_file = os.path.join(project_root, 'dataset_1', 'data', 'healthcare-dataset-stroke-data.csv')
X_train, X_test, y_train, y_test = load_and_preprocess_data(data_file)

# Build neural network model
try:
    nn_model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_initializer=GlorotUniform(seed=42)),
        Dense(32, activation='relu', kernel_initializer=GlorotUniform(seed=42)),
        Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform(seed=42))
    ])
except Exception as e:
    raise RuntimeError(f"Error building neural network model: {e}")

# Configure early stopping
early_stopping = EarlyStopping(monitor='val_recall', patience=5, mode='max', restore_best_weights=True)

# Compile model
try:
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Recall'])
except Exception as e:
    raise RuntimeError(f"Error compiling the model: {e}")

# Train model
try:
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
except tf.errors.InvalidArgumentError as e:
    raise ValueError(f"Invalid arguments in training model: {e}")
except Exception as e:
    raise RuntimeError(f"Unexpected error during model training: {e}")

# Predict and evaluate
try:
    y_pred_proba = nn_model.predict(X_test)
    y_pred = (y_pred_proba > 0.3).astype(int)

    print(classification_report(y_test, y_pred))
except Exception as e:
    raise RuntimeError(f"Error in prediction or evaluation: {e}")

# Plot confusion matrix
try:
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize per row
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap="turbo",
                xticklabels=["Non-Stroke", "Stroke"],
                yticklabels=["Non-Stroke", "Stroke"])
    plt.title("DS 1 NN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
except Exception as e:
    raise RuntimeError(f"Error generating confusion matrix plot: {e}")

import sys
import os
sys.path.append(os.path.abspath(\
    os.path.join(os.path.dirname(__file__), '../../../')))
import random
import numpy as np
import yaml
import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '../../'))

if not os.path.basename(project_root) == 'Assignment-1':
    raise FileNotFoundError('Project root "Assignment-1" not found, please ensure that you are using the specified file structure in the github repo')

yaml_path = os.path.join(project_root,'dataset_2','config.yaml')
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(filepath):
    """this takes the filepath and performs a series of \n
    preprossing only for dataset 2.

    step 1: load dataset into DataFrame.
    step 2: convert cat variables into num using one-hot encoding.
    step 3: uses standard scaler to encode selected features.
    step 4: seperates features and target variable (heart disease).
    step 5: split data into test and train.

    returns train and test set.
    """
    df = pd.read_csv(filepath)

    df_encoded = pd.get_dummies(df, columns=["Sex","ChestPainType",
                                             "RestingECG","ExerciseAngina",
                                             "ST_Slope"],drop_first=True)

    scaler = StandardScaler()
    df_encoded[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = \
        scaler.fit_transform(df_encoded[['Age', 'RestingBP', 'Cholesterol', 
                                     'MaxHR', 'Oldpeak']])

    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

data_file = os.path.join(project_root,'dataset_2','data', 'heart.csv')
X_train, X_test, y_train, y_test = \
    load_and_preprocess_data(data_file)

# Train and evaluate SVM using parameters from config file
svm = SVC(kernel='linear',
          probability=True,
          random_state=42)
s= time.time()
svm.fit(X_train, y_train)
training_time = time.time() - s
print(f"SVM Training Time: {training_time:.4f} seconds")


y_pred_proba = svm.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > .45).astype(int)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cm_percentage_row = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize per row
labels = ["No Heart Disease", "Heart Disease"]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage_row, annot=True,
            fmt='.2f', cmap='turbo', xticklabels=labels,
            yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'DS2 SVM Confusion Matrix (Row-wise Percent)')
plt.show()

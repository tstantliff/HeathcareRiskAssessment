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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE

current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '../../'))

if not os.path.basename(project_root) == 'Assignment-1':
    raise FileNotFoundError('Project root "Assignment-1" not found, please ensure that you are using the specified file structure in the github repo')

yaml_path = os.path.join(project_root,'dataset_1','config.yaml')
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)



def load_and_preprocess_data(file_path):
    """This takes the filepath and performs a series of preprocessing only for dataset 1.

    step 1: Load dataset into DataFrame.
    step 2: Label encode binary categorical variables.
    step 3: One-hot encode categorical features with more than two categories.
    step 4: Separate features and target variable (stroke).
    step 5: Apply Borderline-SMOTE to balance the classes.
    step 6: Split data into test and train.
    step 7: Standardize features

    Returns train and test sets.
    """
    df = pd.read_csv(file_path)

    l = LabelEncoder()
    df['gender'] = l.fit_transform(df['gender'])
    df['ever_married'] = l.fit_transform(df['ever_married'])
    df['Residence_type'] = l.fit_transform(df['Residence_type'])
    
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)
    df['bmi'].fillna(df['bmi'].median(), inplace=True)

    X = df.drop(columns=['stroke', 'id'])
    y = df['stroke']

    b = BorderlineSMOTE(random_state=42)
    X_resampled, y_resampled = b.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2,
        stratify=y_resampled, random_state=42
    )

    s = StandardScaler()
    X_train_scaled = s.fit_transform(X_train)
    X_test_scaled = s.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


data_file = os.path.join(project_root,'dataset_1','data', 'healthcare-dataset-stroke-data.csv')
X_train, X_test, y_train, y_test = \
    load_and_preprocess_data(data_file)

svm = SVC(kernel='linear', probability=True, random_state=42)
start = time.time()
svm.fit(X_train, y_train)
training_time = time.time() - start
print(f"SVM Training Time: {training_time:.4f} seconds")

y_pred_proba = svm.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > .45).astype(int)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize per row
labels = ["No Stroke", "Stroke"]
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'DS1 SVM Confusion Matrix (Row-wise Percent)')
plt.show()

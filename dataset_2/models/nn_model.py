import sys
import os

import random
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '../../'))

if not os.path.basename(project_root) == 'Assignment-1':
    raise FileNotFoundError('Project root "Assignment-1" not found, please ensure that you are using the specified file structure in the github repo')

yaml_path = os.path.join(project_root,'dataset_2','models', 'config_nn.yaml')
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

model = Sequential()
for l in config['neural_network']['layers']:
    model.add(Dense(units=l['units'],
                    activation=l['activation'],
                    input_dim=X_train.shape[1])
                    )
    model.add(BatchNormalization())
    model.add(Dropout(l['dropout']))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=config['neural_network']['learning_rate']),
              loss=config['neural_network']['loss'], 
              metrics=config['neural_network']['metrics'])

early_stopping = EarlyStopping(monitor=config['neural_network']['early_stopping']['monitor'],
                               patience=config['neural_network']['early_stopping']['patience'],
                               restore_best_weights=config['neural_network']['early_stopping']['restore_best_weights'])

s = time.time()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    epochs=config['neural_network']['epochs'], 
                    batch_size=config['neural_network']['batch_size'],
                    callbacks=[early_stopping])

training_time = time.time() - s
print(f"Training Time: {training_time:.4f} seconds")

model.save('models/dataset_2/nn_model_2.h5')

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cm= cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize per row
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="turbo",
            xticklabels=["No Heart Disease", "Heart Disease"],
            yticklabels=["No Heart Disease", "Heart Disease"])
plt.title("DS2 NN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

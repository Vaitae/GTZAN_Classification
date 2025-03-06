# -*- coding: utf-8 -*-
"""gtzan.ipynb

Vaishnavi 

Original file is located at
    https://colab.research.google.com/drive/1M4zEPqx55xXDnUtMW3be-zkg2UY5KlAc
"""

!pip install pandas sklearn pydub librosa
!pip install resampy

import os
from google.colab import drive, files
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

drive.mount('/content/drive')

os.chdir('/content/drive/MyDrive/gtzan_music/Data/genres_original')

def extract_features(file_path):
    try:
        audio_data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while processing {file_path}: {str(e)}")
        return None

data = []
labels = []

# Iterate through the main folder and its subfolders
for root, dirs, files in os.walk('/content/drive/MyDrive/gtzan_music/Data/genres_original'):
    for file_name in files:
        if file_name.endswith('.wav'):
            file_path = os.path.join(root, file_name)
            features = extract_features(file_path)
            if features is not None:
                data.append(features)
                labels.append(file_name.split('.')[0])  # Assuming the genre is in the filename
            else:
                print(f"Skipping {file_path} due to errors during feature extraction.")

# Create a DataFrame from the extracted features and labels
if data:
    df = pd.DataFrame(data, columns=[f'mfcc_{i}' for i in range(1, 41)])
    df['label'] = labels
else:
    print("No audio files found or all files failed during feature extraction.")

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def plot_roc_curve(model, X_test, y_true):
    # Binarize the true labels
    y_true_bin = label_binarize(y_true, classes=model.classes_)

    # Obtain predicted probabilities for each class
    y_pred_proba = model.predict_proba(X_test)

    # Compute the false positive rate, true positive rate, and thresholds for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(model.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curves for each class
    plt.figure()
    for i in range(len(model.classes_)):
        plt.plot(fpr[i], tpr[i], label='Class %d (AUC = %0.2f)' % (i, roc_auc[i]))

    # Set the plot labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Multiclass)')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

# KNN Classifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred)
knn_classification_report = classification_report(y_test, y_pred)
print(f'Accuracy: {knn_accuracy}')
print(f'Classification Report:\n{knn_classification_report}')
knn_cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plot_roc_curve(model, X_test, y_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, y_pred)
logistic_classification_report = classification_report(y_test, y_pred)
print(f'Accuracy: {logistic_accuracy}')
print(f'Classification Report:\n{logistic_classification_report}')
logreg_cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plot_roc_curve(model, X_test, y_test)

# Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)
rf_classification_report = classification_report(y_test, y_pred)
print(f'Accuracy: {rf_accuracy}')
print(f'Classification Report:\n{rf_classification_report}')
rf_cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plot_roc_curve(model, X_test, y_test)

#Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
decision_accuracy = accuracy_score(y_test, y_pred)
decision_classification_report = classification_report(y_test, y_pred)
print(f'Accuracy: {decision_accuracy}')
print(f'Classification Report:\n{decision_classification_report}')
dt_cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plot_roc_curve(model, X_test, y_test)

#Support Vector Machines
model = SVC(probability=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred)
svm_classification_report = classification_report(y_test, y_pred)
print(f'Accuracy: {svm_accuracy}')
print(f'Classification Report:\n{svm_classification_report}')
svm_cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plot_roc_curve(model, X_test, y_test)

#Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred)
nb_classification_report = classification_report(y_test, y_pred)
print(f'Accuracy: {nb_accuracy}')
print(f'Classification Report:\n{nb_classification_report}')
nb_cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plot_roc_curve(model, X_test, y_test)

#Comparing all algorithms
accuracies = [0.565, 0.6, 0.375, 0.435, 0.495, 0.495]
algorithms = ['Logistic Regression', 'Random Forest','Decision Tree','SVM', 'KNN', 'Naive Bayes']

plt.figure(figsize=(8, 6))
plt.bar(algorithms, accuracies)
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Music Genre Prediction Algorithms')
plt.xticks(rotation=45)
plt.ylim([0, 1])
plt.show()


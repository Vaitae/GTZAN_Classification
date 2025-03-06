# GTZAN_Classification

This project evaluates and compares six machine learning models for classifying music genres from the **GTZAN dataset**. It extracts relevant audio features such as **MFCCs (Mel-Frequency Cepstral Coefficients)** using **Librosa** and applies various classifiers to predict the music genre. The models are evaluated based on accuracy and other performance metrics such as precision, recall, and F1-score.

## File Content

1. ``` gtzan.py ```
* Description: Jupyter notebook where the entire process is implemented. This includes:
  - Audio file loading and feature extraction using **Librosa**.
  - Model training and evaluation using six different classifiers (KNN, Logistic Regression, Random Forest, Decision Tree, SVM, and Naive Bayes).
  - Performance comparison of the models using metrics such as accuracy, classification report, confusion matrix, and ROC curve.

## Dataset

This project uses a condensed version of the standard GTZAN dataset, which consists of 1,000 audio tracks divided into 10 music genres (100 tracks per genre). The dataset can be downloaded from the following link:

[Download GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

For best results, it is recommended to use the **official GTZAN dataset**, which provides a more comprehensive set of audio files for training and evaluation.

  ## Requirements
- Python 3.x
- Jupyter Notebook
- Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `librosa`, etc.

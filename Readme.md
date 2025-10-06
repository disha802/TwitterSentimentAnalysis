📘 Overview

This project implements a small-scale replication of the paper “Sentiment Analysis of Twitter Data Using Logistic Regression and Support Vector Machines” (Apeksha Saxena & Disha Kataria, 2024), extending it with a hybrid SVM–Naive Bayes (BFTAN-inspired) approach.

The model classifies tweets as positive or negative, combining statistical text features (TF-IDF) with lexicon-based sentiment scores. It demonstrates how traditional ML and simple lexicon-based heuristics can jointly enhance accuracy on limited data.

🧩 Key Features

Dataset Flexibility: Works with any Twitter-style dataset (twitter_training.csv, Sentiment140, etc.).
Automatic Column Detection: Script identifies text and sentiment columns automatically.
Preprocessing Pipeline: Cleans tweets (removes URLs, hashtags, mentions, punctuation, digits, extra spaces).
Lexicon-based Sentiment Features: Calculates polarity scores using curated positive/negative word lists.
TF-IDF Vectorization: Converts cleaned text into numerical features for ML models.
Hybrid Ensemble:
Linear SVM (high precision on linear separable data)
Multinomial Naive Bayes (robust for noisy text)
Combines both predictions using confidence-weighted voting.
Performance Metrics: Accuracy, Precision, Recall, and Confusion Matrix.
Sample Predictions: Demonstrates real-time classification on test examples.

⚙️ Requirements

Install dependencies (Python 3.8+):
pip install pandas numpy scikit-learn scipy

📂 Project Structure
ML_Sentiment_Analysis/
│
├── twitter_training.csv         # Input dataset (e.g., Sentiment140 subset)
├── sentiment_analysis.py        # Main implementation script
├── README.md                    # Project documentation
└── results/                     # (optional) Store metrics, plots, etc.

🧮 How It Works
1️⃣ Load Dataset
Reads the provided CSV, auto-detecting text and sentiment columns.
Filters to binary classes: positive and negative.

2️⃣ Preprocessing
Performs tokenization, lowercasing, punctuation and stopword removal, and space normalization.

3️⃣ Feature Extraction
TF-IDF Vectorization → represents textual importance of words.
Sentiment Score Extraction → calculates a small numeric sentiment score based on positive/negative word counts.

4️⃣ Model Training
Trains:
Linear Support Vector Classifier (SVM)
Multinomial Naive Bayes (NB)
Combines both using CalibratedClassifierCV (for probability estimates) and a hybrid voting mechanism.

5️⃣ Evaluation
Splits data into 70% train / 30% test, computes:
Accuracy
Precision
Recall
Confusion Matrix

6️⃣ Sample Predictions
Tests a few example sentences to display predicted sentiment labels.

📈 Example Output
Hybrid SVM-BFTAN Sentiment Analysis on Twitter Data

Accuracy:  0.8720 (87.20%)
Precision: 0.8741 (87.41%)
Recall:    0.8698 (86.98%)

Confusion Matrix:
TN: 1295, FP: 186
FN: 204, TP: 1300

Text: This is absolutely fantastic and wonderful!
Prediction: POSITIVE

📚 References
Pang & Lee (2008), Opinion Mining and Sentiment Analysis.
Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python.

🧠 Future Work
Add multiclass support (neutral, sarcasm, mixed).
Expand sentiment lexicon using external dictionaries (e.g., SentiWordNet, VADER).
Integrate deep learning embeddings (Word2Vec, BERT) for semantic context.
Develop a Flask or Streamlit web app for interactive tweet analysis.

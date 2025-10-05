import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

sample_data = [
    ("I love this product! It's amazing", 1),
    ("This is the worst experience ever", -1),
    ("Terrible service, very disappointed", -1),
    ("Excellent quality and fast delivery", 1),
    ("Not good at all, waste of money", -1),
    ("Outstanding performance, highly recommend", 1),
    ("Poor quality, broke after one day", -1),
    ("Best purchase I've made this year", 1),
    ("Horrible customer support", -1),
    ("Fantastic! Exceeded my expectations", 1),
    ("Disappointed with the quality", -1),
    ("Great value for money", 1),
    ("Do not buy this, total scam", -1),
    ("Perfect! Exactly what I needed", 1),
    ("Waste of time and money", -1),
    ("Impressive features and easy to use", 1),
    ("Regret buying this product", -1),
    ("Absolutely love it!", 1),
    ("Defective product, asking for refund", -1),
    ("Superb quality and great price", 1)
]

class TextPreprocessor:
    
    @staticmethod
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class SentimentScoreExtractor:
    
    def __init__(self):
        self.positive_words = {'love', 'great', 'excellent', 'amazing', 'fantastic', 
                              'outstanding', 'perfect', 'best', 'superb', 'impressive'}
        self.negative_words = {'hate', 'terrible', 'worst', 'horrible', 'poor', 
                              'bad', 'disappointed', 'waste', 'defective', 'regret'}
    
    def calculate_sentiment_score(self, text):
        words = text.lower().split()
        positive = sum(1 for word in words if word in self.positive_words)
        negative = sum(1 for word in words if word in self.negative_words)
        
        if positive + negative == 0:
            return 0.0
        
        score = (positive - negative) / (positive + negative + 2)
        return score
    
    def extract_features(self, texts):
        return np.array([self.calculate_sentiment_score(text) for text in texts]).reshape(-1, 1)

class HybridSVMBFTAN:
    
    def __init__(self):
        self.svm_classifier = SVC(kernel='linear', C=1.0, probability=True)
        self.nb_classifier = GaussianNB()
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.sentiment_extractor = SentimentScoreExtractor()
        self.is_fitted = False
    
    def fit(self, X_text, y):
        X_tfidf = self.vectorizer.fit_transform(X_text).toarray()
        X_sentiment = self.sentiment_extractor.extract_features(X_text)
        X_combined = np.hstack([X_tfidf, X_sentiment])
        self.svm_classifier.fit(X_combined, y)
        
        self.nb_classifier.fit(X_combined, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X_text):
        """Predict using hybrid approach"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Extract features
        X_tfidf = self.vectorizer.transform(X_text).toarray()
        X_sentiment = self.sentiment_extractor.extract_features(X_text)
        X_combined = np.hstack([X_tfidf, X_sentiment])
        
        svm_pred = self.svm_classifier.predict(X_combined)
        svm_proba = self.svm_classifier.predict_proba(X_combined)
        
        nb_pred = self.nb_classifier.predict(X_combined)
        nb_proba = self.nb_classifier.predict_proba(X_combined)
        
        hybrid_pred = []
        for i in range(len(X_text)):
            if svm_pred[i] == nb_pred[i]:
                hybrid_pred.append(svm_pred[i])
            else:
                svm_conf = np.max(svm_proba[i])
                nb_conf = np.max(nb_proba[i])
                if svm_conf > nb_conf:
                    hybrid_pred.append(svm_pred[i])
                else:
                    hybrid_pred.append(nb_pred[i])
        
        return np.array(hybrid_pred)

def main():
    print("Hybrid SVM-BFTAN Sentiment Analysis Implementation")
    
    # Prepare data
    texts = [item[0] for item in sample_data]
    labels = [item[1] for item in sample_data]
    
    # Preprocess
    preprocessor = TextPreprocessor()
    processed_texts = [preprocessor.preprocess(text) for text in texts]
    
    print("\nSample preprocessed texts:")
    for i in range(3):
        print(f"Original: {texts[i]}")
        print(f"Processed: {processed_texts[i]}\n")
    
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.3, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")
    
    # Train hybrid model
    print("Training Hybrid SVM-BFTAN model...")
    model = HybridSVMBFTAN()
    model.fit(X_train, y_train)
    print("Training complete!\n")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics (as per equations 11, 12, 13 in paper)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    # Test on new samples
    print("\n" + "=" * 60)
    print("TESTING ON NEW SAMPLES")
    print("=" * 60)
    
    test_samples = [
        "This is absolutely fantastic and wonderful!",
        "Terrible experience, very unhappy with this",
        "Good product with some minor issues"
    ]
    
    processed_test = [preprocessor.preprocess(text) for text in test_samples]
    predictions = model.predict(processed_test)
    
    for text, pred in zip(test_samples, predictions):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment}")
    
    print("\n" + "=" * 60)
    print("Note: With a larger dataset, accuracy would improve significantly.")
    print("The paper achieved 80%+ accuracy with 120K tweets.")
    print("=" * 60)

if __name__ == "__main__":
    main()
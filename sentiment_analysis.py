import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="wide")

# Title
st.title("💬 Hybrid SVM-BFTAN Sentiment Analysis")
st.markdown("---")

# Preprocessing class
class TextPreprocessor:
    @staticmethod
    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# Hybrid Model (no manual sentiment words)
class HybridSVMBFTAN:
    def __init__(self):
        base_svm = LinearSVC(C=1.0, max_iter=1000, dual=False)
        self.svm_classifier = CalibratedClassifierCV(base_svm, cv=3)
        self.nb_classifier = MultinomialNB()
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95, ngram_range=(1, 2))
        self.is_fitted = False
    
    def fit(self, X_text, y):
        X_tfidf = self.vectorizer.fit_transform(X_text)
        
        self.svm_classifier.fit(X_tfidf, y)
        self.nb_classifier.fit(X_tfidf, y)
        self.is_fitted = True
        return self
    
    def predict(self, X_text):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        X_tfidf = self.vectorizer.transform(X_text)
        
        svm_pred = self.svm_classifier.predict(X_tfidf)
        svm_proba = self.svm_classifier.predict_proba(X_tfidf)
        nb_pred = self.nb_classifier.predict(X_tfidf)
        nb_proba = self.nb_classifier.predict_proba(X_tfidf)
        
        hybrid_pred = []
        for i in range(len(X_text)):
            if svm_pred[i] == nb_pred[i]:
                hybrid_pred.append(svm_pred[i])
            else:
                svm_conf = np.max(svm_proba[i])
                nb_conf = np.max(nb_proba[i])
                hybrid_pred.append(svm_pred[i] if svm_conf > nb_conf else nb_pred[i])
        
        return np.array(hybrid_pred)
    
    def predict_proba(self, X_text):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        X_tfidf = self.vectorizer.transform(X_text)
        svm_proba = self.svm_classifier.predict_proba(X_tfidf)
        nb_proba = self.nb_classifier.predict_proba(X_tfidf)
        
        # Average probabilities from both models
        avg_proba = (svm_proba + nb_proba) / 2
        return avg_proba

# Load and train model with caching
@st.cache_resource
def load_and_train_model():
    file_path = "data/twitter_data.csv"
    
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Verify expected columns exist
        if 'clean_text' not in df.columns or 'category' not in df.columns:
            st.error(f"Expected columns 'clean_text' and 'category', but found: {list(df.columns)}")
            raise ValueError("Dataset must have 'clean_text' and 'category' columns")
        
        text_col = 'clean_text'
        sentiment_col = 'category'
        
        # Remove rows with missing values
        df = df.dropna(subset=[text_col, sentiment_col])
        
        # Map sentiments to numeric labels
        if df[sentiment_col].dtype == 'object':
            df[sentiment_col] = df[sentiment_col].str.strip().str.lower()
            df = df[df[sentiment_col].isin(['positive', 'negative'])]
            df["label"] = df[sentiment_col].map({"positive": 1, "negative": -1})
        else:
            unique_vals = sorted(df[sentiment_col].unique())
            df = df[df[sentiment_col].isin([-1, 1])]
            df["label"] = df[sentiment_col]
            
            if len(df) == 0:
                st.error(f"No data found with categories -1 or 1. Found categories: {unique_vals}")
                raise ValueError("No valid positive or negative samples found in dataset")
        
        df = df.dropna(subset=["label"])
        
        if len(df) == 0:
            raise ValueError("No valid data after filtering. Please check your dataset format.")
        
        # Sample data
        sample_size = 10000
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        texts = df[text_col].tolist()
        labels = df["label"].tolist()
        
        # Preprocess
        preprocessor = TextPreprocessor()
        processed_texts = [preprocessor.preprocess(text) for text in texts]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Train model
        model = HybridSVMBFTAN()
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
        recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return model, preprocessor, metrics
        
    except FileNotFoundError:
        st.error(f"Dataset file not found at: {file_path}")
        st.info("Please ensure your dataset is located at 'data/twitter_data.csv'")
        raise
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        raise

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    
    if st.button("Train/Load Model", type="primary"):
        with st.spinner("Loading and training model..."):
            try:
                model, preprocessor, metrics = load_and_train_model()
                st.session_state['model'] = model
                st.session_state['preprocessor'] = preprocessor
                st.session_state['metrics'] = metrics
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
    
    st.markdown("---")
    
    if 'metrics' in st.session_state:
        st.subheader("Model Performance")
        metrics = st.session_state['metrics']
        
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("Precision", f"{metrics['precision']:.2%}")
        st.metric("Recall", f"{metrics['recall']:.2%}")
        
        st.caption(f"Training samples: {metrics['train_size']}")
        st.caption(f"Testing samples: {metrics['test_size']}")
        
        st.subheader("Confusion Matrix")
        cm = metrics['confusion_matrix']
        st.write(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
        st.write(f"FN: {cm[1][0]} | TP: {cm[1][1]}")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Analyze Text")
    
    # Text input
    user_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    # Analyze button
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        elif 'model' not in st.session_state:
            st.error("❌ Please train/load the model first using the sidebar button.")
        else:
            model = st.session_state['model']
            preprocessor = st.session_state['preprocessor']
            
            with st.spinner("Analyzing..."):
                processed = preprocessor.preprocess(user_input)
                prediction = model.predict([processed])[0]
                probabilities = model.predict_proba([processed])[0]
                
                st.session_state['last_prediction'] = prediction
                st.session_state['last_text'] = user_input
                st.session_state['last_probabilities'] = probabilities
    
    # Display result
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.subheader("Result")
        
        prediction = st.session_state['last_prediction']
        probabilities = st.session_state.get('last_probabilities', None)
        
        if prediction == 1:
            st.success("😊 POSITIVE SENTIMENT")
        else:
            st.error("😞 NEGATIVE SENTIMENT")
        
        if probabilities is not None:
            # probabilities[0] is negative class, probabilities[1] is positive class
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            st.progress(float(confidence))
            st.caption(f"Confidence: {confidence:.2%}")
        
        st.info(f"**Analyzed Text:** {st.session_state['last_text']}")

with col2:
    st.header("📝 Quick Examples")
    
    examples = [
        "This is absolutely fantastic and wonderful!",
        "Terrible experience, very poor service",
        "Good product with some minor issues",
        "I love this! Best purchase ever!",
        "Worst service I've ever encountered"
    ]
    
    for example in examples:
        if st.button(example, key=example, use_container_width=True):
            if 'model' not in st.session_state:
                st.error("Please train/load the model first.")
            else:
                model = st.session_state['model']
                preprocessor = st.session_state['preprocessor']
                
                processed = preprocessor.preprocess(example)
                prediction = model.predict([processed])[0]
                probabilities = model.predict_proba([processed])[0]
                
                st.session_state['last_prediction'] = prediction
                st.session_state['last_text'] = example
                st.session_state['last_probabilities'] = probabilities
                st.rerun()

st.markdown("---")
st.caption("Hybrid SVM-BFTAN Sentiment Analysis System | Built with 💙 by Apeksha and Disha")

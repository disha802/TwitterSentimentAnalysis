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

# Sentiment Score Extractor
class SentimentScoreExtractor:
    def __init__(self):
        self.positive_words = {'love', 'great', 'excellent', 'amazing', 'fantastic', 
                              'outstanding', 'perfect', 'best', 'superb', 'impressive',
                              'good', 'wonderful', 'awesome', 'brilliant', 'happy'}
        self.negative_words = {'hate', 'terrible', 'worst', 'horrible', 'poor', 
                              'bad', 'disappointed', 'waste', 'defective', 'regret',
                              'awful', 'useless', 'sad', 'angry', 'disgusting'}
    
    def calculate_sentiment_score(self, text):
        words = text.lower().split()
        positive = sum(1 for word in words if word in self.positive_words)
        negative = sum(1 for word in words if word in self.negative_words)
        
        if positive + negative == 0:
            return 0.0
        score = (positive - negative) / (positive + negative + 2)
        score = (score + 1) / 2
        return score
    
    def extract_features(self, texts):
        return np.array([self.calculate_sentiment_score(text) for text in texts]).reshape(-1, 1)

# Hybrid Model
class HybridSVMBFTAN:
    def __init__(self):
        base_svm = LinearSVC(C=1.0, max_iter=1000, dual=False)
        self.svm_classifier = CalibratedClassifierCV(base_svm, cv=3)
        self.nb_classifier = MultinomialNB()
        self.vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.95)
        self.sentiment_extractor = SentimentScoreExtractor()
        self.is_fitted = False
    
    def fit(self, X_text, y):
        X_tfidf = self.vectorizer.fit_transform(X_text)
        X_sentiment = self.sentiment_extractor.extract_features(X_text)
        X_sentiment_sparse = csr_matrix(X_sentiment)
        X_combined = hstack([X_tfidf, X_sentiment_sparse])
        
        self.svm_classifier.fit(X_combined, y)
        self.nb_classifier.fit(X_combined, y)
        self.is_fitted = True
        return self
    
    def predict(self, X_text):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        X_tfidf = self.vectorizer.transform(X_text)
        X_sentiment = self.sentiment_extractor.extract_features(X_text)
        X_sentiment_sparse = csr_matrix(X_sentiment)
        X_combined = hstack([X_tfidf, X_sentiment_sparse])
        
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
                hybrid_pred.append(svm_pred[i] if svm_conf > nb_conf else nb_pred[i])
        
        return np.array(hybrid_pred)

# Load and train model with caching
@st.cache_resource
def load_and_train_model():
    file_path = "data/twitter_training.csv"
    
    try:
        df = pd.read_csv(file_path)
        text_col = None
        sentiment_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'text' in col_lower or 'tweet' in col_lower:
                text_col = col
            if 'sentiment' in col_lower or 'label' in col_lower or 'target' in col_lower:
                sentiment_col = col
        
        if text_col is None or sentiment_col is None:
            if len(df.columns) >= 4:
                text_col = df.columns[3]
                sentiment_col = df.columns[2]
            else:
                text_col = df.columns[-1]
                sentiment_col = df.columns[0]
    except:
        df = pd.read_csv(file_path, header=None)
        df.columns = ["id", "entity", "sentiment", "tweet"]
        text_col = "tweet"
        sentiment_col = "sentiment"
    
    # Filter and map sentiments
    if df[sentiment_col].dtype == 'object':
        df = df[df[sentiment_col].str.lower().isin(['positive', 'negative'])]
        df["label"] = df[sentiment_col].str.lower().map({"positive": 1, "negative": -1})
    else:
        unique_vals = sorted(df[sentiment_col].unique())
        if 4 in unique_vals:
            df = df[df[sentiment_col].isin([0, 4])]
            df["label"] = df[sentiment_col].map({0: -1, 4: 1})
        else:
            df = df[df[sentiment_col].isin([0, 1])]
            df["label"] = df[sentiment_col].map({0: -1, 1: 1})
    
    sample_size = 10000
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    texts = df[text_col].tolist()
    labels = df["label"].tolist()
    
    # Preprocess
    preprocessor = TextPreprocessor()
    processed_texts = [preprocessor.preprocess(text) for text in texts]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.3, random_state=42
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
        'confusion_matrix': cm
    }
    
    return model, preprocessor, metrics

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    
    if st.button("Train/Load Model", type="primary"):
        with st.spinner("Loading and training model..."):
            model, preprocessor, metrics = load_and_train_model()
            st.session_state['model'] = model
            st.session_state['preprocessor'] = preprocessor
            st.session_state['metrics'] = metrics
            st.success("Model loaded successfully!")
    
    st.markdown("---")
    
    if 'metrics' in st.session_state:
        st.subheader("Model Performance")
        metrics = st.session_state['metrics']
        
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("Precision", f"{metrics['precision']:.2%}")
        st.metric("Recall", f"{metrics['recall']:.2%}")
        
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
            st.warning("Please enter some text to analyze.")
        elif 'model' not in st.session_state:
            st.error("Please train/load the model first using the sidebar button.")
        else:
            model = st.session_state['model']
            preprocessor = st.session_state['preprocessor']
            
            with st.spinner("Analyzing..."):
                processed = preprocessor.preprocess(user_input)
                prediction = model.predict([processed])[0]
                
                st.session_state['last_prediction'] = prediction
                st.session_state['last_text'] = user_input
    
    # Display result
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.subheader("Result")
        
        prediction = st.session_state['last_prediction']
        
        if prediction == 1:
            st.success("😊 POSITIVE SENTIMENT")
        else:
            st.error("😞 NEGATIVE SENTIMENT")
        
        st.info(f"**Analyzed Text:** {st.session_state['last_text']}")

with col2:
    st.header("📝 Quick Examples")
    
    examples = [
        "This is absolutely fantastic and wonderful!",
        "Terrible experience, very unhappy with this",
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
                
                st.session_state['last_prediction'] = prediction
                st.session_state['last_text'] = example
                st.rerun()

st.markdown("---")
st.caption("Hybrid SVM-BFTAN Sentiment Analysis System | Built with 💙 by Apeksha and Disha")
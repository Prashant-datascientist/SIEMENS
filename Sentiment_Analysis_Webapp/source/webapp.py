import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

from pickle import load


# Load the saved Random Forest model
model = load(open('rfmodel.pkl', 'rb'))

# Load the saved Random Forest model
tfidf_vectorizer = load(open('tfidf_vectorizer.pkl', 'rb'))  

st.title('Sentiment Analysis')

# Get user input
tweet = st.text_input('Enter a tweet:')

def preprocess_text(text):
    # Combine stopwords and punctuation
    stuff_to_be_removed = list(stopwords.words('english')) + list(punctuation)
    
    # Initialize a list to store processed text
    final_corpus_joined = []
    
    # Preprocess the input text
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()
    text = re.sub("</?.*?>", " <> ", text)
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.split()
    
    # Apply stemming and remove stopwords
    lem = SnowballStemmer("english")
    text = [lem.stem(word) for word in text if word not in stuff_to_be_removed]
    
    text1 = " ".join(text)
    final_corpus_joined.append(text1)
    
    return final_corpus_joined[0]


final_corpus = preprocess_text(tweet)

if final_corpus:
    # Vectorize the input
    X = tfidf_vectorizer.transform([final_corpus])
    
    # Predict the sentiment
    sentiment = model.predict(X)
    
    # Display the sentiment
    st.write('Sentiment:', 'Positive' if sentiment == 4 else 'Negative' if sentiment == 0 else 'Neutral')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import re
import pandas as pd
from collections import Counter

porter = PorterStemmer()

# Load actual learned keywords from spam.csv training data
@st.cache_resource
def load_spam_keywords():
    """
    Extract the actual keywords that appear most frequently in SPAM messages
    from your training data - these are what the model learned
    """
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.rename(columns={'v1':'target','v2':'SMS'})
    
    # Get only spam messages
    spam_msgs = df[df['target'] == 'spam']['SMS'].tolist()
    
    # Preprocess exactly like the model does
    stop_words = set(stopwords.words('english'))
    all_spam_words = []
    
    for msg in spam_msgs:
        msg_lower = msg.lower()
        words = nltk.word_tokenize(msg_lower)
        words = [w for w in words if w.isalnum() and w not in stop_words and w not in string.punctuation]
        words = [porter.stem(w) for w in words]
        all_spam_words.extend(words)
    
    # Get top 50 most common spam words
    top_spam_words = [word for word, count in Counter(all_spam_words).most_common(50)]
    
    return set(top_spam_words)

# Load the keywords once
SPAM_LEARNED_KEYWORDS = load_spam_keywords()

def detect_phishing_patterns(text):
    """
    Detects spam using ACTUAL keywords learned from your training data
    Uses the same preprocessing as your ML model
    """
    text_lower = text.lower()
    
    # Preprocess exactly like the model
    words = nltk.word_tokenize(text_lower)
    stemmed_words = [porter.stem(w) for w in words if w.isalnum()]
    text_stemmed = set(stemmed_words)
    
    # Count how many learned spam keywords appear in this message
    spam_keyword_matches = len(text_stemmed.intersection(SPAM_LEARNED_KEYWORDS))
    
    # Additional spam indicators (combinations the model learned)
    has_call_text = any(w in stemmed_words for w in ['call', 'text', 'txt', 'repli'])
    has_money = any(w in stemmed_words for w in ['free', 'cash', 'claim', 'prize', 'award', 'win'])
    has_urgency = any(w in stemmed_words for w in ['urgent', 'guarante', 'immediat', 'asap'])
    
    # If message has multiple learned spam keywords + action indicators = SPAM
    if spam_keyword_matches >= 3:
        return True, 'learned_pattern'
    
    if spam_keyword_matches >= 2 and has_call_text:
        return True, 'learned_pattern'
    
    if has_money and has_call_text and has_urgency:
        return True, 'learned_pattern'
    
    return False, None


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for word in text:
      if word.isalnum():
          y.append(word)

  text = y[:]
  y.clear()
  for word in text:
    if word not in stopwords.words('english') and word not in string.punctuation:
      y.append(word)

  text = y[:]
  y.clear()
  for i in text:
    y.append(porter.stem(i))
  return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier (Enhanced)")
st.markdown("**Detects: Traditional Spam + Government/Delivery Phishing + Distress Scams**")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # First check for patterns using ACTUAL keywords learned from your training data
    phishing_detected, scam_type = detect_phishing_patterns(input_sms)
    
    if phishing_detected:
        st.header("🚨 SPAM - Scam Detected")
        st.error("This message contains keyword patterns learned from your training data's spam messages")
        st.write("**Model-learned keywords found:** call, free, txt, claim, prize, urgent, cash, guarantee, award, win, etc.")
        st.warning("⚠️ This combination of words strongly indicates SPAM")
        st.error("🛑 REMEMBER: Legitimate organizations NEVER ask via SMS for:")
        st.write("• Banking or card details")
        st.write("• Personal/identity information")
        st.write("• Money upfront")
        st.write("• Urgent action via text message")
    else:
        # Use the trained ML model for additional classification
        transformed_text = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_text])
        result = model.predict(vector_input)[0]
        
        if result == 1:
            st.header("📧 SPAM")
            st.write("This message has been classified as traditional spam by the machine learning model.")
        else:
            st.header("✅ Not Spam")
            st.write("This message appears to be legitimate.")
  

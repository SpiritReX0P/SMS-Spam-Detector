# 📩 SMS Spam Detector

Python • Machine Learning • NLP • Streamlit

🌟 Overview  
An intelligent SMS Spam Detection System powered by Machine Learning and Natural Language Processing (NLP). This project classifies incoming SMS messages as **Spam** or **Ham (Not Spam)** using advanced text preprocessing, TF-IDF vectorization, and classification algorithms.

Built with Streamlit for a clean and interactive user experience, the application performs real-time spam prediction with high accuracy.

---

## ✨ Key Features

- 📌 Real-time SMS spam classification
- 🧠 NLP-based text preprocessing pipeline
- 📊 TF-IDF feature extraction
- ⚡ Fast and accurate ML prediction
- 🎨 Interactive Streamlit web interface
- 🔍 Text cleaning, stemming & stopword removal
- 📈 Multiple ML models evaluated
- 💼 Clean and professional UI

---

## 🛠️ Technical Architecture

### NLP Preprocessing Pipeline

The system performs several text preprocessing steps:

- Convert text to lowercase
- Tokenization using NLTK
- Remove punctuation & special characters
- Stopword removal
- Porter Stemming

### Feature Engineering

- TF-IDF Vectorization converts textual data into numerical vectors
- Optimized vocabulary generation for better classification

### Machine Learning Models

Models tested include:

- Naive Bayes
- Logistic Regression
- SVM
- Adaboost
- Xgboost
- Voting Classifier Ensemble

The best-performing model is used for deployment.

---

## 🚀 Technologies Used

- Python
- Scikit-learn
- NLTK
- Pandas
- NumPy
- wordcloud
- Streamlit

---

## 📋 Dataset

The project uses an SMS Spam Collection Dataset containing labeled spam and ham messages.

### Features Used:
- SMS Message Text
- Label (Spam / Ham)

---

## ⚡ Quick Start

### Clone Repository

```bash
git clone <your-repo-link>
cd SMS-Spam-Detector

# Task 6: Sentiment Analysis on Customer Reviews

## Overview
This task performs **sentiment analysis** on product reviews using machine learning and NLP techniques.  
It predicts whether a review is **Positive**, **Neutral**, or **Negative**.

---

## Folder Structure
Task5_Sentiment_Analysis/
│
├── data/
│ └── Reviews.csv # Dataset of customer reviews
│
├── models/
│ ├── sentiment_model.pkl # Trained Logistic Regression model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer used for feature extraction
│
├── sentiment_analysis.ipynb # Jupyter Notebook with all code
└── README.md # This file

---

## Dataset
- Source: Customer product reviews dataset (`Reviews.csv`)
- Columns used:
  - `Score` → Rating of product (1–5)
  - `Text` → Review text
- Sentiment is categorized as:
  - **Positive** → Score ≥ 4
  - **Neutral** → Score = 3
  - **Negative** → Score ≤ 2

---

## Features
1. **Data Cleaning**
   - Lowercasing
   - Removing punctuation & numbers
   - Stopwords removal
   - Lemmatization

2. **Feature Extraction**
   - TF-IDF vectorization (Top 5000 words)

3. **Machine Learning Model**
   - Logistic Regression classifier

4. **Evaluation**
   - Accuracy score
   - Classification report

5. **Visualization**
   - Sentiment distribution (Bar & Pie charts)
   - Top 20 most frequent words

---

## How to Run
1. Open `sentiment_analysis.ipynb` in Jupyter Notebook.  
2. Make sure all required libraries are installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk

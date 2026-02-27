# 📌 Requirement Classification using NLP

## Overview

This project implements a Natural Language Processing (NLP) pipeline to automatically classify software requirements into predefined categories such as functional, usability, performance, and security.

The goal is to streamline requirement analysis by transforming raw textual requirement descriptions into structured, categorized outputs using classical machine learning techniques.

---

## Dataset

The dataset consists of software requirements collected from different projects. Each entry contains:

* **ProjectID**
* **RequirementText** (natural language description)
* **Class label** (e.g., F, US, PE, SE, LF, etc.)

These labels represent requirement categories such as functional, usability, performance, and security.

---

## Methodology

The system follows a traditional NLP pipeline:

### 1️⃣ Text Preprocessing

* Lowercasing
* Removing punctuation and numbers
* Stopword removal (including domain-specific words like “shall”, “must”)
* Tokenization (NLTK)
* Lemmatization (WordNetLemmatizer)

### 2️⃣ Feature Extraction

* Bag-of-Words (BoW) representation using `CountVectorizer`

### 3️⃣ Handling Class Imbalance

* SMOTE (Synthetic Minority Over-sampling Technique)

### 4️⃣ Model Training

* Logistic Regression
* Hyperparameter tuning with GridSearchCV
* Train-test split (70%-30%)

---

## Results

The Logistic Regression model achieved:

* **68% accuracy** on the test dataset
* Precision, Recall, and F1-score evaluated per class

This demonstrates that classical NLP techniques can provide a solid baseline for automated requirement classification tasks.

---

## Limitations

* Bag-of-Words does not capture semantic relationships between words.
* Logistic Regression may not generalize well for more complex datasets.

---

## Future Improvements

* TF-IDF or Word Embeddings (Word2Vec, GloVe)
* Transformer-based models (e.g., BERT)
* Deep learning approaches (LSTM, Transformer architectures)


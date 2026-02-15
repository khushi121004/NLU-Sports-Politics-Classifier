# Sports vs Politics Text Classification
Course: Natural Language Understanding  
Assignment 1 – Problem 4  
Student: Agrawal Khushi Jitendra (B22CS005)

## Overview
This project implements a binary text classification system that classifies news articles as:
- Sport
- Politics

The dataset is derived from the BBC News dataset, filtered to include only these two categories.

## Feature Extraction
Although Bag of Words and TF-IDF were studied, the final implementation uses:

N-grams (ngram_range = (1,2))

This captures unigrams and bigrams such as:
- "prime minister"
- "world cup"

## Models Used
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

## Results
- Naive Bayes: 99.46%
- Logistic Regression: 98.38%
- SVM: 100%

## Requirements
Python 3.x  
pandas  
numpy  
scikit-learn  

Install dependencies:
pip install pandas numpy scikit-learn

## How to Run
Make sure the dataset file is present, then run:

python main.py

The script will:
- Preprocess the text
- Extract N-gram features
- Train models
- Print evaluation results

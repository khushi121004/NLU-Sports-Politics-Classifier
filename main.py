import pandas as pd
from preprocess import clean_text
from feature_extraction import ngram_features
from train_models import split_data, train_nb, train_lr, train_svm
from evaluate import evaluate

# Load data
data = pd.read_csv("sports_politics_dataset.csv")

# Clean text
data["cleaned"] = data["text"].apply(clean_text)

# Split raw text first
X_train_text, X_test_text, y_train, y_test = split_data(data["cleaned"], data["category"])

# Fit on train only
X_train, vectorizer = ngram_features(X_train_text)

# Transform test using SAME vectorizer
X_test = vectorizer.transform(X_test_text)

print("Training Naive Bayes")
nb = train_nb(X_train, y_train)
evaluate(nb, X_test, y_test)

print("\nTraining Logistic Regression")
lr = train_lr(X_train, y_train)
evaluate(lr, X_test, y_test)

print("\nTraining SVM")
svm = train_svm(X_train, y_train)
evaluate(svm, X_test, y_test)

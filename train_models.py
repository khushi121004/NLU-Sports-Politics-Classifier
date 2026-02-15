from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def split_data(X, y): #80-20 train test split
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_nb(X_train, y_train): #Model 1-Multinomial Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_lr(X_train, y_train): #Model 2-Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train): #Model 3-Linear SVC
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model

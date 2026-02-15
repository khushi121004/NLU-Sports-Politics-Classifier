from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def bow_features(texts): #Bag of Words features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts) #learn vocabulary and transform text into matrix
    return X, vectorizer

def tfidf_features(texts): #TF-IDF features
    vectorizer = TfidfVectorizer() #create TF-IDF Vectorizer object
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def ngram_features(texts): #N-gram features
    vectorizer = CountVectorizer(ngram_range=(1,2))  # uses unigram + bigram features hence range(1,2)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

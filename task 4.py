import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
from collections import Counter
from nltk.stem.snowball import SnowballStemmer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# Load the spam dataset
spam_data = pd.read_csv('D:/Internships/codesoft/task 4/spam.csv', encoding='latin', usecols=['v1', 'v2'])
spam_data.columns = ['label', 'text']
spam_data.drop_duplicates(inplace=True)
spam_data['label'] = spam_data['label'].map({'spam': 1, 'ham': 0})

# Visualize the distribution of label
num_ham, num_spam = len(spam_data[spam_data["label"] == 0]), len(spam_data[spam_data["label"] == 1])
label_names = np.array(["Non-Spam", "Spam"])
label_counts = np.array([num_ham, num_spam])
plt.figure(figsize=(5, 5))
plt.pie(label_counts, labels=label_names, autopct="%.1f%%", colors=['Red', '#1E90FF'])
plt.show()

stop_words = set(stopwords.words('english'))

# Data cleaning and preprocessing
def text_preprocessing(text):
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    stemmer = SnowballStemmer("english")
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

spam_data['text'] = spam_data['text'].apply(text_preprocessing)

X = spam_data['text']
y = spam_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43, stratify=y)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Count Vectorization
count_vectorizer = CountVectorizer(ngram_range=(1, 3))
X_train_cv = count_vectorizer.fit_transform(X_train)
X_test_cv = count_vectorizer.transform(X_test)

# Model training and evaluation with TF-IDF Vectorization
mnb = MultinomialNB()
params = {'alpha': [0.1, 0.5, 0.8, 1, 2, 5, 7, 10]}
randomized_search_tfidf = RandomizedSearchCV(mnb, params, scoring='accuracy', cv=10, n_jobs=-1, random_state=3, verbose=3)
randomized_search_tfidf.fit(X_train_tfidf, y_train)
print('Best Score (TF-IDF) is ', randomized_search_tfidf.best_score_, ' for ', randomized_search_tfidf.best_params_)

mnb_tfidf = MultinomialNB(alpha=0.5)
mnb_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = mnb_tfidf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_tfidf))

cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
fig_tfidf, ax_tfidf = plot_confusion_matrix(conf_mat=cm_tfidf, show_absolute=True, show_normed=True, colorbar=True, class_names=["Non-Spam", "Spam"], cmap='Blues')

plt.show()

# Model training and evaluation with Count Vectorization
randomized_search_cv = RandomizedSearchCV(mnb, params, scoring='accuracy', cv=10, n_jobs=-1, random_state=3, verbose=3)
randomized_search_cv.fit(X_train_cv, y_train)
print('Best Score (Count Vectorizer) is ', randomized_search_cv.best_score_, ' for ', randomized_search_cv.best_params_)

mnb_cv = MultinomialNB(alpha=5)
mnb_cv.fit(X_train_cv, y_train)
y_pred_cv = mnb_cv.predict(X_test_cv)
print(classification_report(y_test, y_pred_cv))


cm_cv = confusion_matrix(y_test, y_pred_cv)
fig_cv, ax_cv = plot_confusion_matrix(conf_mat=cm_cv, show_absolute=True, show_normed=True, colorbar=True, class_names=["Non-Spam", "Spam"], cmap='Blues')

plt.show()

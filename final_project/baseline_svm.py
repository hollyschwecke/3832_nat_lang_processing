import numpy as np 
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from utilities import clean_tweet, print_metrics, plot_confusion_matrix

# load dataset
dataset = load_dataset("tweet_eval", "irony")
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']
test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

# clean tweets
train_texts = [clean_tweet(t) for t in train_texts]
test_texts = [clean_tweet(t) for t in test_texts]

# TF-IDF + SVM pipeline
pipline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LinearSVC(class_weight='balanced'))
])

# train and evaluate 
pipline.fit(train_texts, train_labels)
preds = pipline.predict(test_texts)

# print metrics
print_metrics(test_labels, preds, target_names=["Not Sarcastic", "Sarcastic"])

# plot confusion matrix
plot_confusion_matrix(test_labels, preds, labels=["Not Sarcastic", "Sarcastic"])
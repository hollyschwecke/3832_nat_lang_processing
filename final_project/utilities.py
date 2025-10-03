import re
import numpy as np 
import random
import torch
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------
# text cleaning for tweets
# ------------------------------------------------
def clean_tweet(text):
    '''
    cleans a tweet by removing username, hashtags, URLs, emojis, punctuation, and lowercasing
    '''
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'@\w+', '', text) # remove mentions
    text = re.sub(r'#\w+', '', text) # remove hashtags
    text = re.sub(r'http\S+', '', text) # remove URLs
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # remove extra whitespace
    return text
    
# ------------------------------------------------
# classification metrics summary
# ------------------------------------------------
def print_metrics(true_labels, predictions, target_names=None):
    '''
    prints classification metrics reports 
    '''
    print(classification_report(true_labels, predictions, digits=4, target_names=target_names))
    
# ------------------------------------------------
# confusion matrix plot
# ------------------------------------------------
def plot_confusion_matrix(true_labels, predictions, labels=None, title="Confusion Matrix"):
    '''
    plots a confusion matrix using seaborn heatmap
    '''
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
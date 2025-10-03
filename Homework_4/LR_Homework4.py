
# In the chapter on logistic regression, the book suggests a very small number
# of features for classifying movie reviews
# x1: count of positive words in the document
# x2: count of negative words in the document
# x3: 1 if "no" is in document, 0 otherwise
# x4 count of first and second person pronouns
# x5 1 if "!" is in document, 0 otherwise
# x6 log(word count of document)

# let's see if it works with Stochastic Gradient Descent
import string

from nltk.corpus import movie_reviews
import random
import nltk
import math
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


# the following block of code creates a dictionary of sentiment
# words whose value is 1 if it is a positive word
# and 0 if it is a negative word.
sentimentWordDictionary = {}
f = open('positive-words.txt', 'r', encoding="ISO-8859-1")
for line in f:
    line = line.strip()
    if len(line) == 0: # ignore this line
        continue
    if line[0] == ';': # ignore this line
        continue
    sentimentWordDictionary[line.lower()] = 1
f.close()
f = open('negative-words.txt', 'r', encoding="ISO-8859-1")
for line in f:
    line = line.strip()
    if len(line) == 0: # ignore this line
        continue
    if line[0] == ';': # ignore this line
        continue
    sentimentWordDictionary[line.lower()] = 0
f.close()

# for debugging purposes
print("There are", len(sentimentWordDictionary), "sentiment words.")

# Grab all the documents and shuffle them
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
print("There are", len(documents), "documents.") # good for debugging


def countPositiveAndNegativeWords(d):
    '''
    Counts the number of positive and negative words in the document
    :param d: A list containing the words in the document
    :return: A tuple (positive, negative) with two integer values representing
            the number of positive and negative words
    '''
    countPositive = 0
    countNegative = 0

    # you have to do this
    
    for word in d:
        w = word.lower()
        if w in sentimentWordDictionary:
            if sentimentWordDictionary[w] == 1:
                countPositive += 1
            elif sentimentWordDictionary[w] == 0:
                countNegative += 1
    return countPositive, countNegative


def noInDocument(d):
    '''
    Returns 1 if the word "no" is in the document.   You may
    want to contemplate whether you need to make this case sensitive or not.
    :param d: A list of words in the document.
    :return: 1 if no is in document; 0, otherwise.
    '''

    # you have to do this
    return 1 if "no" in [word.lower() for word in d] else 0

def countFirstSecondPersonPronouns(d):
    '''
    Returns a count of the number of first and second person pronouns
    within the document.  You might want to look up what is a first or second
    person pronoun.
    :param d: A list of words in the document.
    :return: The count of personal pronouns
    '''
    count = 0
    # complete this
    
    pronouns = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours'}
    for word in d:
        if word.lower() in pronouns:
            count += 1
    return count

def exclamationInDocument(d):
    '''
    Returns 1 if the word "!" is in the document.
    :param d: A list of words in the document.
    :return: 1 if ! is in document; 0, otherwise.
    '''

    # complete this
    return 1 if "!" in d else 0

def logOfLength(d):
    '''
    Computes and returns the log of the number of tokens in the document.
    :param d: A list of words in the document.
    :return: log(number of words)
    '''
    # complete this
    return math.log(len(d) + 1) # add 1 to avoid log(0)

def sentimentWordRatio(d):
    '''
    Computes the ratio of positive - negative words to total words
    :param d: A list of words in the document
    :return: sentiment word ratio'''
    pos, neg = countPositiveAndNegativeWords(d)
    total = len(d)
    if total == 0: # neutral sentiment
        return 0
    # positive return value = positive sentiment
    # negative return value = negative sentiment
    return (pos - neg) / total 


def document_features(document):
    '''
    Builds the set of features for each document.
    You don't need to modify this unless
    you want to add another feature.
    :param document: A list of words in the document.
    :return: A dictionary containing the features for that document.
    '''
    document_words = list(document) # do not turn into a set!!
    features = {}
    positive, negative = countPositiveAndNegativeWords(document_words)

    features['positiveCount']  = positive
    features['negativeCount'] = negative
    features['noInDoc'] = noInDocument(document_words)
    features['personalPronounCount'] = countFirstSecondPersonPronouns(document_words)
    features['exclamation'] = exclamationInDocument(document_words)
    features['logLength'] = logOfLength(document_words)
    features['sentimentWordRatio'] = sentimentWordRatio(document_words)

    return features

# for each document, extract its features
featuresets = [(document_features(d), c) for (d,c) in documents]

# build the training and test sets
trainingSize = int(0.8*len(featuresets))
train_set, test_set = featuresets[0:trainingSize], featuresets[trainingSize:]

# use stochastic gradient descent with log loss function
classifier = LogisticRegression(max_iter=1000, verbose=0)
x_Train = [list(a.values()) for (a,b) in train_set]
y_Train = [b for (a,b) in train_set]
classifier.fit(x_Train, y_Train)

# print(classifier.coef_)  # if you want to see the coefficients, unsorted

x_Test = [list(a.values()) for (a,b) in test_set]
y_Test = [b for (a,b) in test_set]

print("LR Fit", classifier.score(x_Test, y_Test))

# here is a block of code that sorts the features by absolute value
# and prints them out
featureNames = ['positiveCount', 'negativeCount', 'noInDoc', 'personalPronounCount',  'exclamation', 'logLength', 'sentimentWordRatio']
featuresPlusImportance = [ (featureNames[i], classifier.coef_[0][i]) for i in range(len(classifier.coef_[0]))]
featuresPlusImportance.sort(key = lambda x: abs(x[1]), reverse=True)
for x in range(len(featuresPlusImportance)):
    print(featuresPlusImportance[x])


correct_tags = [c for (w, c) in test_set]
test_tags = list(classifier.predict(x_Test))

# how about its precision and recall per category
mtrx = nltk.ConfusionMatrix(correct_tags, test_tags)
print()
print(mtrx)
print()
print(mtrx.evaluate())


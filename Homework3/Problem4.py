import random
import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.classify import apply_features
from nltk import FreqDist, NaiveBayesClassifier, classify, DecisionTreeClassifier, MaxentClassifier
from nltk.metrics import precision, recall
from collections import defaultdict

nltk.download("movie_reviews")
nltk.download("stopwords")

# load and shuffle documents
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# build vocab
all_words = FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

# feature extractor
def document_features(document, word_feats=word_features):
    document_words = set(document)
    return {f'contains({word})': (word in document_words) for word in word_feats}

# metrics calculation
def evaluate_classifier(classifier, test_set):
    accuracy = classify.accuracy(classifier, test_set)
    
    # build reference and test sets for precision/recall
    refsets = defaultdict(set)
    testsets = defaultdict(set)
    
    for i, (features, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = classifier.classify(features)
        testsets[observed].add(i)
        
    precision_score = {label: precision(refsets[label], testsets[label]) for label in refsets}
    recall_scores = {label: recall(refsets[label], testsets[label]) for label in refsets}
    
    return accuracy, precision_score, recall_scores

# universal training pipeline
def train_and_evaluate(feature_func, documents, label=""):
    featuresets = [(feature_func(d), c) for (d, c) in documents]
    train_set, test_set = featuresets[100:], featuresets[:100]
    
    classifiers = {
        "Naive Bayes": NaiveBayesClassifier.train(train_set),
        "Decision Tree": DecisionTreeClassifier.train(train_set),
        "MaxEnt (IIS)": MaxentClassifier.train(train_set, algorithm="iis", trace=0, max_iter=10)
    }
    
    print(f"\m=== Results for {label} ===")
    for name, clf in classifiers.items():
        accuracy, precision_scores, recall_scores = evaluate_classifier(clf, test_set)
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision_scores}")
        print(f"Recall: {recall_scores}")
        if name == "Naive Bayes":
            clf.show_most_informative_features(5)
            
# b) feature extractor with stopword removal
stop_words = set(stopwords.words('english'))
def document_features_no_stopwords(document):
    filtered = [w for w in document if w.lower() not in stop_words]
    document_words = set(filtered)
    return {f'contains({word})': (word in document_words) for word in word_features}

# c) negation-based feature extractor
def negate_sequence(document):
    result = []
    negate = False
    for word in document:
        lower = word.lower()
        if lower == 'not':
            result.append('not')
            negate = True
        elif negate:
            result.append(f'not_{lower}')
        else:
            result.append(lower)
    return result

def document_features_with_negation(document):
    modified_doc = negate_sequence(document)
    document_words = set(modified_doc)
    return {f'contains({word})': (word in document_words) for word in word_features}

# a) baseline (no preprocessing)
train_and_evaluate(document_features, documents, "Original")

# b) with stopword removal
train_and_evaluate(document_features_no_stopwords, documents, "No Stopwords")

# c) with negation modification
train_and_evaluate(document_features_with_negation, documents, "Negation Handling")
import nltk

nltk.download('all')

from nltk.book import *
from nltk import FreqDist

# list of all words in text1, converted to lowercase
lowercase_words = [word.lower() for word in text1]

# filter the list to include only words with 10 or more characters
long_words = [word for word in lowercase_words if len(word) >= 10]

# build a frequency distribution using nltk.FreqDist
f_dist = FreqDist(long_words)

# get five most common long words
common_long_words = f_dist.most_common(5)

print("Most common 10+ character words in Moby Dick:")
for word, count in common_long_words:
    print(f"{word}: {count}")
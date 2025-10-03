import nltk
from nltk import word_tokenize, bigrams, FreqDist
import os

# download necessary nltk tokenizer
nltk.download("punkt")

# read in text file
file_path = os.path.join(os.path.dirname(__file__), "DorianGray.txt")
with open(file_path, 'r', encoding="utf-8") as file:
    text = file.read()

# tokenize text into words
tokens = word_tokenize(text)

# a) 5 most frequent bigrams
bigrams_list = list(bigrams(tokens)) # generate bigrams
bigram_freq = FreqDist(bigrams_list) # count bigram frequency
most_common_bigrams = bigram_freq.most_common(5)

print("a) Five most frequent birgrams:")
for bg in most_common_bigrams:
    print(f"{bg[0]} -> {bg[1]} times")
    
# b) Five most common words after "the" or "The"
following_words = []

for i in range(len(tokens) - 1):
    if tokens[i].lower() == 'the':
        following_words.append(tokens[i + 1])

following_freq = FreqDist(following_words)
most_common_following = following_freq.most_common(5)

print("\nb) Five most frequent words after 'the' or 'The':")
for word, freq in most_common_following:
    print(f"{word} -> {freq} times")
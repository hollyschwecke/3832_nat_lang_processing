import nltk
from nltk.metrics import edit_distance
from nltk.tokenize import word_tokenize

# download necessary nltk tokenizer
nltk.download('punkt')

# a) character level edit distance

str1 = "INTENTION"
str2 = "EXECUTION"

# default sub cost is 1, Jurafsky and Martin use sub cost 2
distance_chars = edit_distance(str1, str2, substitution_cost=2)
print("a) Edit distance between 'INTENTION' and 'EXECUTION':", distance_chars)

# b) word level edit distance

sent1 = "The girl hit the ball"
sent2 = "The girl danced at the ball"

# use nltk.word_tokenize to get lists of words
words1 = word_tokenize(sent1)
words2 = word_tokenize(sent2)

print("Tokenized sentence 1:", words1)
print("Tokenized sentence 2:", words2)

# compute edit distance between word lists
distance_words = edit_distance(words1, words2)
print("b) Word-level edit distance between the two sentences:", distance_words)
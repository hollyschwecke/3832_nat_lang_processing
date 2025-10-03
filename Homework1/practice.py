from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer

word = "humanity"

stemmer_l = LancasterStemmer()
stemmer_p = PorterStemmer()

print(stemmer_l.stem(word))
print(stemmer_p.stem(word))

import gensim.downloader as api
import nltk
from sklearn.metrics.pairwise import cosine_similarity

listOfExamples = [
    ("apple", "banana", "pear", "broccoli"),
    ("dog", "cat", "bird", "chair"),
    ("red", "green", "blue", "circle"),
    ("hammer", "screwdriver", "nail", "pen"),
    ("mountain", "river", "ocean", "book"),
    ("Monday", "Tuesday", "July", "Thursday"),
    ("lion", "tiger", "bear", "shark"),
    ("table", "chair", "bed", "computer"),
    ("football", "basketball", "chess", "baseball"),
    ("Paris", "London", "New_York", "Africa"),
    ("gold", "silver", "water", "bronze"),
    ("run", "jump", "dance", "chair"),
    ("spring", "summer", "hot", "winter"),
    ("triangle", "circle", "square", "potato"),
    ("car", "train", "plane", "dog"),
    ("math", "science", "history", "pizza"),
    ("eagle", "sparrow", "crow", "dolphin"),
    ("pencil", "eraser", "notebook", "shoes"),
    ("Earth", "Mars", "Saturn", "Sun"),
    ("ice", "fire", "snow", "water"),
    ('sneaker', 'sock', 'blouse', 'sandal'),
    ('tattoo', 'necklace', 'bracelet', 'earring'),
    ('doctor', 'teacher', 'engineer', 'carpenter'), # all jobs, but carpenter is manual label
    ('drums', 'guitar', 'bass', 'violin'), # 3 are string instruments, violin is not likely to be used by a band (ie rock band)
    ('water', 'ice', 'juice', 'lemonade'),
    ('river', 'bank', 'tributary', 'stream'),
    ('bat', 'ball', 'glove', 'club')
    
]



# Load pre-trained Word2Vec model (Google's Word2Vec)
model = api.load('word2vec-google-news-300')
# Made around 2013 from Google News Dataset.  About 100 billion words.
# The size of the vector is 300.
# The date range of the news articles is mid-2000s to 2013.

def word_in_model(word):
    return word in model

def average_similarity(model, w1, w2, w3):
    '''
    Computes the average cosine similarity between
    w1 and w2, w1 and w3, and w2 and w3.
    :param model: The word2vec model passed in.
    :param w1: Word 1.  A string.
    :param w2: Word 2.  A string.
    :param w3: Word 3.  A string.
    :return: A floating point number. The average cosine similarity.
    '''
    # needs to be completed

    # here's an example of how to compute the cosine similarity between w1 and w2
    vector1 = model[w1]
    vector2 = model[w2]
    vector3 = model[w3]
    sim_1_2 = cosine_similarity([vector1], [vector2])[0][0]
    sim_1_3 = cosine_similarity([vector1], [vector3])[0][0]
    sim_2_3 = cosine_similarity([vector2], [vector3])[0][0]
    average = (sim_1_2 + sim_1_3 + sim_2_3) / 3
    return average

def find_odd_one_out(example):
    '''
    Finds the odd one out of a given example.
    :param example: A tuple of four words (strings)
    :return: The odd one out.  A string.
    '''
    (a, b, c, d) = example  # get the four words
    # needs to be completed
    
    # try removing each word and compute the average similarity of the remaining 3
    try:
        sim_if_remove_a = average_similarity(model, b, c, d)
        sim_if_remove_b = average_similarity(model, a, c, d)
        sim_if_remove_c = average_similarity(model, a, b, d)
        sim_if_remove_d = average_similarity(model, a, b, c)
    # skip words not in model
    except:
        return "Unknown"
    
    similarities = {
        a: sim_if_remove_a,
        b: sim_if_remove_b,
        c: sim_if_remove_c,
        d: sim_if_remove_d
    }
    oddOneOut = max(similarities, key=similarities.get)
    return oddOneOut

for example in listOfExamples:
    oddOneOut = find_odd_one_out(example)
    print(oddOneOut, "is the odd one out of group", example )



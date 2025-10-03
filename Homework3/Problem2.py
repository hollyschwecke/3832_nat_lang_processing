from nltk.corpus import wordnet as wn

word_pairs = [
    ("car", "automobile"),
    ("gem", "jewel"),
    ("journey", "voyage"),
    ("boy", "lad"),
    ("coast", "shore"),
    ("asylum", "madhouse"),
    ("magician", "wizard"),
    ("midday", "noon"),
    ("furnace", "stove"),
    ("food", "fruit"),
    ("bird", "cock"),
    ("bird", "crane"),
    ("tool", "implement"),
    ("brother", "monk"),
    ("lad", "brother"),
    ("crane", "implement"),
    ("journey", "car"),
    ("monk", "oracle"),
    ("cemetery", "woodland"),
    ("food", "rooster"),
    ("coast", "hill"),
    ("forest", "graveyard"),
    ("shore", "woodland"),
    ("monk", "slave"),
    ("coast", "forest"),
    ("lad", "wizard"),
    ("chord", "smile"),
    ("glass", "magician"),
    ("rooster", "voyage"),
    ("noon", "string"),
]

# helper function to get first synset of words
def get_first_synset(word):
    synsets = wn.synsets(word)
    return synsets[0] if synsets else None

# compute the similarity between all word pairs
def compute_similarities(pairs):
    results = []
    for w1, w2 in pairs:
        # get first synset of each word
        syn1 = get_first_synset(w1)
        syn2 = get_first_synset(w2)
        
        if syn1 and syn2:
            # print(f"{w1} -> {syn1}, {w2} -> {syn2}") # check to make sure get_first_synset() works
            sim = syn1.wup_similarity(syn2)
            # print(f"Similarity: {sim}") # check to make sure sim works
            similarity = sim if sim is not None else 0
        else:
            similarity = 0
        results.append(((w1, w2), similarity))
    return sorted(results, key=lambda x: x[1], reverse=True)

def main():
    similarities = compute_similarities(word_pairs)
    print("Similarity scores ranked (highest to lowest):\n")
    for (w1, w2), score in similarities:
        print(f'{w1:10s} - {w2:10s}: {score:.3f}')
        
if __name__ == "__main__":
    main()
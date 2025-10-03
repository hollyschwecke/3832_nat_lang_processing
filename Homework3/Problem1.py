from nltk.corpus import wordnet as wn

def compute_average_polysemy(pos):
    ''' 
    computes average polysemy for a given part of speech
    param pos: part of speech ('n', 'v', 'a', 'r')
    return: average number of senses per word (polysemy)
    '''
    word_sense_counts = {}
    for synset in wn.all_synsets(pos):
        for lemma in synset.lemmas():
            name = lemma.name()
            if name not in word_sense_counts:
                word_sense_counts[name] = len(wn.synsets(name, pos))
    if not word_sense_counts:
        return 0
    total_senses = sum(word_sense_counts.values())
    return total_senses / len(word_sense_counts)

def main():
    pos_labels = {'n': 'Noun', 'v': 'Verb', 'a': 'Adjective', 'r': 'Adverb'}
    for pos, label in pos_labels.items():
        avg_polysemy = compute_average_polysemy(pos)
        print(f'Average polysemy of {label}s: {avg_polysemy:.2f}')
        
if __name__ == '__main__':
    main()
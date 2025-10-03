from nltk.corpus import brown
from collections import defaultdict, Counter

def most_common_plurals(tagged_words):
    # extract singular and plural regular nouns
    singular_nouns = [w.lower() for (w, p) in tagged_words if p == 'NN']
    plural_nouns = [w.lower() for (w, p) in tagged_words if p == 'NNS']
    
    # count occurrences
    singular_counts = Counter(singular_nouns)
    plural_counts = Counter(plural_nouns)
    
    more_common_plural = []
    
    # compare regular plural nouns ending in 's' to their singular forms
    for word in plural_counts:
        if word.endswith('s'):
            singular_form = word[:-1]
            if singular_counts[singular_form] < plural_counts[word]:
                more_common_plural.append((word, plural_counts[word], singular_counts[singular_form]))
    return sorted(more_common_plural, key=lambda x: x[1] - x[2], reverse=True)

def word_with_most_tags(tagged_words):
    tag_dict = defaultdict(set)
    
    # build a dictionary mapping each word to the set of tags it has
    for word, tag in tagged_words:
        tag_dict[word.lower()].add(tag)
        
    # find the word with the highest number of distinct tags
    max_tag_word = max(tag_dict.items(), key=lambda x: len(x[1]))
    
    return max_tag_word[0], max_tag_word[1]

def common_tags_before_nouns(tagged_words):
    tag_before_noun = defaultdict(int)
    
    for i in range(1, len(tagged_words)):
        word, tag = tagged_words[i]
        if 'NN' in tag:
            prev_tag = tagged_words[i - 1][1]
            tag_before_noun[prev_tag] += 1
    # sort tags by count desc
    return sorted(tag_before_noun.items(), key=lambda x: x[1], reverse=True)

def main():
    # use brown corpus
    tagged_words = brown.tagged_words()
    
    # question 1: plural nouns more common than singular
    print("1. Nouns that are more common in plural form:")
    plural_results = most_common_plurals(tagged_words)
    for word, plural_count, singular_count in plural_results[:10]:
        print(f"{word}: plural= {plural_count}, singular = {singular_count}")
    print()
    
    # question 2: word with the most distinct tags
    print("2. Word with the greatest number of distinct tags:")
    word, tags = word_with_most_tags(tagged_words)
    print(f"{word} -> {tags} (Total tags: {len(tags)})")
    print()
    
    # question 3: most common tags before nouns
    print("3. Tags most commonly found before nouns:")
    common_tags = common_tags_before_nouns(tagged_words)
    for tag, count in common_tags[:10]:
        print(f"{tag}: {count}")
        
if __name__ == "__main__":
    main()
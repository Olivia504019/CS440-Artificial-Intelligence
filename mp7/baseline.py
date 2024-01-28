from collections import defaultdict

"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_tag_cnts = defaultdict(lambda: defaultdict(int))
    word_best_tag = {}
    tag_cnts = defaultdict(int)
    predicts = []
    most_seen_tag = None
    for sentence in train:
        for word, tag in sentence:
            word_tag_cnts[word][tag] += 1
            tag_cnts[tag] += 1
    
    for tag in tag_cnts.keys():
        if not most_seen_tag or tag_cnts[tag] > tag_cnts[most_seen_tag]:
            most_seen_tag = tag

    for word in word_tag_cnts.keys():
        for tag in word_tag_cnts[word].keys():
            if word not in word_best_tag or word_tag_cnts[word][tag] > word_tag_cnts[word][word_best_tag[word]]:
                word_best_tag[word] = tag

    for sentence in test:
        predict = []
        for word in sentence:
            if word not in word_best_tag:
                predict.append((word, most_seen_tag))
            else:
                predict.append((word, word_best_tag[word]))
        predicts.append(predict)
    return predicts
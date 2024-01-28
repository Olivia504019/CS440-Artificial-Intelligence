"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: math.log(emit_epsilon))) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: math.log(epsilon_for_pt))) # {tag0:{tag1: # }}
    tags = set()
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    tag_trans_cnts = {}
    emit_cnts = {}
    for sentence in sentences:
        prev_tag = None
        for word, curr_tag in sentence:
            if curr_tag not in tags:
                tags.add(curr_tag)
            if prev_tag:
                if prev_tag not in tag_trans_cnts:
                    tag_trans_cnts[prev_tag] = {}
                if curr_tag not in tag_trans_cnts[prev_tag]:
                    tag_trans_cnts[prev_tag][curr_tag] = 0
                tag_trans_cnts[prev_tag][curr_tag] += 1
            prev_tag = curr_tag
            if curr_tag not in emit_cnts:
                emit_cnts[curr_tag] = {}
            if word not in emit_cnts[curr_tag]:
                emit_cnts[curr_tag][word] = 0
            emit_cnts[curr_tag][word] += 1
            prev_tag = curr_tag

    for prev_tag in tag_trans_cnts.keys():
        for curr_tag in tags:
            if curr_tag in tag_trans_cnts[prev_tag]:
                tag_trans_cnts[prev_tag][curr_tag] += epsilon_for_pt
            else:
                tag_trans_cnts[prev_tag][curr_tag] = epsilon_for_pt

    for prev_tag in tag_trans_cnts.keys():
        for curr_tag in tag_trans_cnts[prev_tag].keys():
            trans_prob[prev_tag][curr_tag] = math.log(tag_trans_cnts[prev_tag][curr_tag]) - math.log(sum(tag_trans_cnts[prev_tag].values()))

    init_prob["START"] = 1

    for tag in emit_cnts.keys():
        for word in emit_cnts[tag].keys():
            emit_prob[tag][word] = math.log(emit_cnts[tag][word] + emit_epsilon) - math.log(sum(emit_cnts[tag].values()) + emit_epsilon * (len(emit_cnts[tag]) + 1))
        emit_prob[tag]["UNKNOWN"] = math.log( emit_epsilon) - math.log(sum(emit_cnts[tag].values()) + emit_epsilon * (len(emit_cnts[tag]) + 1))
        

    return init_prob, emit_prob, trans_prob, tags

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob, tags):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = defaultdict(lambda: 0) # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    if i == 0:
        return log_prob, predict_tag_seq
    else:
        for curr_tag in tags:
            for prev_tag in tags:
                if word in emit_prob[curr_tag]:
                    curr_log_prob = prev_prob[prev_tag] + trans_prob[prev_tag][curr_tag] + emit_prob[curr_tag][word]
                else:
                    curr_log_prob = prev_prob[prev_tag] + trans_prob[prev_tag][curr_tag] + emit_prob[curr_tag]["UNKNOWN"]
                if curr_tag not in log_prob or log_prob[curr_tag] < curr_log_prob:
                    log_prob[curr_tag] = curr_log_prob
                    predict_tag_seq[curr_tag] = prev_tag
    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob, tags = get_probs(train)
    
    """
    for tag in trans_prob.keys():
        print(tag, ': ')
        for ntag in trans_prob[tag].keys():
            print('\t', ntag, ':', trans_prob[tag][ntag])
        for word in emit_prob[tag].keys():
            print('\t', word, ':', emit_prob[tag][word])
    """
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        total_predict_tag_seq = []
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob, tags)
            total_predict_tag_seq.append(predict_tag_seq)
        # break
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        predict = []
        curr_tag = "END"

        for i in range(length - 1, -1, -1):
            predict.append((sentence[i], curr_tag))
            if i > 0:
                curr_tag = total_predict_tag_seq[i][curr_tag]
        predict.reverse()
        predicts.append(predict)

    return predicts
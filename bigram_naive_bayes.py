# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter, defaultdict


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=1.0, pos_prior=0.5, silently=False):
    train_P_cnts = [defaultdict(int), defaultdict(int)]
    train_data_length = len(train_set)
    train_total_word_cnts = [0, 0]
    train_words_set = set()
    yhats = []
    log_P = [1, 1]
    
    # negative = 0, positive = 1 
    # P(word|Type = negative) = counts of word when Type = negative / total negative words counts
    #                                     = train_P_cnts[0] / train_total_word_cnts[0]
    # P(word|Type = positive) = counts of word when Type = positive / total positive words counts
    #                                     = train_P_cnts[1] / train_total_word_cnts[1]
    # train_P_cnts[0]: counts of word when Type = negative
    # train_P_cnts[1]: counts of word when Type = positive
    # train_total_word_cnts[0]: counts of total negative words
    # train_total_word_cnts[1]: counts of total positive words
 
    for i in range(train_data_length):
        for word in train_set[i]:
            train_P_cnts[train_labels[i]][word] += 1
            train_words_set.add(word)
        train_total_word_cnts[train_labels[i]] += len(train_set[i])

    for doc in tqdm(dev_set, disable=silently):
        log_P[0], log_P[1] = math.log(1 - pos_prior), math.log(pos_prior)
        for label in [0, 1]:
            for word in doc:
                log_P[label] += (1 - bigram_lambda)*(math.log(train_P_cnts[label][word] + unigram_laplace) - math.log(train_total_word_cnts[label] + unigram_laplace * len(train_words_set) + 1)) 
                + bigram_lambda*(math.log(train_P_cnts[label][word] + bigram_laplace) - math.log(train_total_word_cnts[label] + bigram_laplace * len(train_words_set) + 1))
        if log_P[0] >= log_P[1]:
            yhats.append(0)
        else:
            yhats.append(1)

    return yhats





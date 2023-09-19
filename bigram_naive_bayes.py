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
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels

# Model description
# negative = 0, positive = 1 
# P(word|Type = negative) = counts of word when Type = negative / total negative words counts
#                                     = train_P_cnts[0] / train_total_word_cnts[0]
# P(word|Type = positive) = counts of word when Type = positive / total positive words counts
#                                     = train_P_cnts[1] / train_total_word_cnts[1]
# train_P_cnts[0]: counts of word when Type = negative
# train_P_cnts[1]: counts of word when Type = positive
# train_total_word_cnts[0]: counts of total negative words
# train_total_word_cnts[1]: counts of total positive words

def unigramBayesModel(dev_set, train_set, train_labels, laplace, pos_prior):
    train_P_cnts = [defaultdict(int), defaultdict(int)]
    train_data_length = len(train_set)
    train_total_word_cnts = [0, 0]
    train_words_set = set()
    log_P = [[], []]
 
    for i in range(train_data_length):
        for word in train_set[i]:
            train_P_cnts[train_labels[i]][word] += 1
            train_words_set.add(word)
        train_total_word_cnts[train_labels[i]] += len(train_set[i])

    for doc in tqdm(dev_set):
        log_P[0].append(math.log(1 - pos_prior))
        log_P[1].append(math.log(pos_prior))
        for label in [0, 1]:
            for word in doc:
                log_P[label][-1] += math.log(train_P_cnts[label][word] + laplace) - \
                                    math.log(train_total_word_cnts[label] + laplace * (len(train_words_set) + 1))
    return log_P

def bigramBayesModel(dev_set, train_set, train_labels, laplace, pos_prior):
    train_P_cnts = [defaultdict(int), defaultdict(int)]
    train_data_length = len(train_set)
    train_total_word_cnts = [0, 0]
    train_words_set = set()
    log_P = [[], []]
 
    for i in range(train_data_length):
        for j in range(len(train_set[i]) - 1):
            word_pair = train_set[i][j] + "," + train_set[i][j + 1]
            train_P_cnts[train_labels[i]][word_pair] += 1
            train_words_set.add(word_pair)
        train_total_word_cnts[train_labels[i]] += len(train_set[i]) - 1

    for doc in tqdm(dev_set):
        log_P[0].append(math.log(1 - pos_prior))
        log_P[1].append(math.log(pos_prior))
        for label in [0, 1]:
            for i in range(len(doc) - 1):
                word_pair = doc[i] + "," + doc[i + 1]
                log_P[label][-1] += math.log(train_P_cnts[label][word_pair] + laplace) - \
                                    math.log(train_total_word_cnts[label] + laplace * (len(train_words_set) + 1))

    return log_P

"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.1, bigram_laplace=0.9, bigram_lambda=0.6, pos_prior=0.8, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    yhats = []
    log_P = [0, 0]
    log_P_unigram = unigramBayesModel(dev_set, train_set, train_labels, unigram_laplace, pos_prior)
    log_P_bigram =  bigramBayesModel(dev_set, train_set, train_labels, bigram_laplace, pos_prior)
    for i in range(len(log_P_unigram[0])):
        for label in [0, 1]:
            log_P[label] = (1 - bigram_lambda) * log_P_unigram[label][i] + bigram_lambda * log_P_bigram[label][i]
        if log_P[0] >= log_P[1]:
            yhats.append(0)
        else:
            yhats.append(1)
    return yhats



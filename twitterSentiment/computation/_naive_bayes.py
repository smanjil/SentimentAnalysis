
from __future__ import division
from config import debug
from cosine_similarity import CosineSimilarity
from testing_tf_idf import TestingTfIdf

class NaiveBayes:
    # constructor imports cosine similarities from cosine_similarity.py
    def __init__(self):
        cs = CosineSimilarity()
        testing = TestingTfIdf()

        cs.separate_positive_and_negative_vectors()
        cs.calculate_product_cosines()
        testing.calculate_tf_idf()

        self.positive_train_tfidf = cs.positive_train_tfidf
        self.negative_train_tfidf = cs.negative_train_tfidf
        self.test_tf_idf = testing.tot_tf_idf
        self.product_pos_cos_sim = cs.product_pos_cos_sim
        self.product_neg_cos_sim = cs.product_neg_cos_sim
        
        # debug program
        if debug:
            print 'Testing: \n', self.test_tf_idf
            print '\nPositive training tf-idf: \n', self.positive_train_tfidf, '\n'
            print '\nNegative training tf-idf: \n', self.negative_train_tfidf, '\n'
            print len(self.positive_train_tfidf), len(self.positive_train_tfidf), len(self.negative_train_tfidf)

        self.prob_pos_train = format((len(self.positive_train_tfidf) / \
                                (len(self.positive_train_tfidf) + len(self.negative_train_tfidf))), '.3f')
        self.prob_neg_train = format((len(self.negative_train_tfidf) / \
                                (len(self.positive_train_tfidf) + len(self.negative_train_tfidf))), '.3f')
        
        # debug program
        if debug:
            print 'Probability of positive training tweets: ', self.prob_pos_train, '\n'
            print 'Probability of negative training tweets: ', self.prob_neg_train, '\n'

        self.calculate_naive_bayes()
        self.determine_pos_neg_tweets()

    # calculate naive bayes probability
    def calculate_naive_bayes(self):
        # calculate probabilities for positive documents
        self.prob_pos_tweets = []
        # debug program
        if debug:
            print 'Probabilities of positive tweets: '
        for items in self.product_pos_cos_sim:
            self.prob_pos_tweets.append(float(items) * float(self.prob_pos_train))
            # self.prob_pos_tweets.append(format((float(items) * float(self.prob_pos_train)), '.8f'))
        # debug program
        if debug:
            print self.prob_pos_tweets, '\n'

        # calculate probabilities for negative documents
        self.prob_neg_tweets = []
        # debug program
        if debug:
            print 'Probabilities of negative tweets: '
        for items in self.product_neg_cos_sim:
            self.prob_neg_tweets.append(float(items) * float(self.prob_neg_train))
            # self.prob_neg_tweets.append(format((float(items) * float(self.prob_neg_train)), '.8f'))
        # debug program
        if debug:
            print self.prob_neg_tweets, '\n'

    # determine positive or negative tweets
    def determine_pos_neg_tweets(self):
        # debug program
        if debug:
            print 'Tweets Classified: '
        self.classify_tweets = []
        for i in range(len(self.prob_pos_tweets)):
            if self.prob_pos_tweets[i] > self.prob_neg_tweets[i]:
                self.classify_tweets.append("Positive")
            elif self.prob_pos_tweets[i] < self.prob_neg_tweets[i]:
                self.classify_tweets.append("Negative")
            else:
                self.classify_tweets.append("Neutral")
        # debug program
        if debug:
            print self.classify_tweets, '\n '


# nb = NaiveBayes()

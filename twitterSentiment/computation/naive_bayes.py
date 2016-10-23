
from __future__ import division
from config import debug
from cosine_similarity import CosineSimilarity
from testing_tf_idf import TestingTfIdf

class NaiveBayes:
    # constructor imports cosine similarities from cosine_similarity.py
    def __init__(self):
        cs = CosineSimilarity()
        testing = TestingTfIdf()

        self.positive_train_tfidf = cs.positive_train_tfidf
        self.negative_train_tfidf = cs.negative_train_tfidf
        self.test_tf_idf = testing.tot_tf_idf
        self.product_pos_cos_sim = cs.product_pos_cos_sim
        self.product_neg_cos_sim = cs.product_neg_cos_sim

        self.prob_pos_train = format((len(self.positive_train_tfidf) / \
                                (len(self.positive_train_tfidf) + len(self.negative_train_tfidf))), '.3f')
        self.prob_neg_train = format((len(self.negative_train_tfidf) / \
                                (len(self.positive_train_tfidf) + len(self.negative_train_tfidf))), '.3f')

        print '\nNaive Bayes: (naive_bayes.py)\n'

        self.calculate_naive_bayes()
        self.determine_pos_neg_tweets()

    # calculate naive bayes probability
    def calculate_naive_bayes(self):
        self.prob_pos_tweets = (float(items) * float(self.prob_pos_train) for items in self.product_pos_cos_sim)        
        self.prob_neg_tweets = (float(items) * float(self.prob_neg_train) for items in self.product_neg_cos_sim)

    # determine positive or negative tweets
    def determine_pos_neg_tweets(self):
        pos, neg = 0, 0
        print 'Tweets Classified: '
        self.classify_tweets = []
        for x, y in zip(self.prob_pos_tweets, self.prob_neg_tweets):
            if x >= y:
                pos += 1
                self.classify_tweets.append("Positive")
            else:
                neg += 1
                self.classify_tweets.append("Negative")
        if debug:
            print self.classify_tweets, '\n'
        print 'Pos: ', pos, 'Neg: ', neg, '\n'

nb = NaiveBayes()

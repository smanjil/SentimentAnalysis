
from __future__ import division
import math
from config import debug
from naive_bayes import NaiveBayes

class Accuracy:
    def __init__(self):
        nb = NaiveBayes()

        self.positive_tfidf = nb.positive_train_tfidf
        self.negative_tfidf = nb.negative_train_tfidf

        print '\nTotal positive data: ', len(self.positive_tfidf)
        print 'Total negative data: ', len(self.negative_tfidf), '\n'

        self.posit_train_tfidf = [j for i, j in enumerate(self.positive_tfidf) \
                                          if i < int(len(self.positive_tfidf) * 0.7)]
        self.negat_train_tfidf = [j for i, j in enumerate(self.negative_tfidf) \
                                          if i < int(len(self.negative_tfidf) * 0.7)]
        self.total_train_tfidf = self.posit_train_tfidf + self.negat_train_tfidf

        print '\nTotal positive training data: ', len(self.posit_train_tfidf)
        print 'Total negative training data: ', len(self.negat_train_tfidf)
        print 'Total training data: ', len(self.total_train_tfidf), '\n'


        self.pos_test = [j for i, j in enumerate(self.positive_tfidf) \
                                          if i >= int(len(self.positive_tfidf) * 0.7)]
        self.neg_test = [j for i, j in enumerate(self.negative_tfidf) \
                                          if i >= int(len(self.negative_tfidf) * 0.7)]
        self.total_test = self.pos_test + self.neg_test

        print '\nTotal positive testing data: ', len(self.pos_test)
        print 'Total negative testing data: ', len(self.neg_test)
        print 'Total testing data: ', len(self.total_test), '\n'

        self.calculate_dot_product()
        self.calculate_magnitude()
        self.calculate_cosine_similarity()
        self.calculate_product_cosines()
        self.calculate_naive_bayes()
        self.determine_pos_neg_tweets()

    ############################### calculate dot product #####################################

    def calculate_dot_product(self):
        self.pos_dot_product, self.neg_dot_product = [], []

        if debug:
            print '\nPositive dot products: '
        for items in self.total_test:
            pos_dot = []
            for item in self.posit_train_tfidf:
                sum_dot_product = 0
                for vec in item[0]:
                    if vec in items[0]:
                        acc_dot_product = item[0][vec] * items[0][vec]
                        if acc_dot_product == 0.0:
                            acc_dot_product = 0.001
                        sum_dot_product += acc_dot_product
                pos_dot.append(format(sum_dot_product, '.3f'))
            self.pos_dot_product.append(pos_dot)
        if debug:
            print '\nLength: ', len(self.pos_dot_product), '\n', self.pos_dot_product

        if debug:
            print '\nNegative dot products: '
        for items in self.total_test:
            neg_dot = []
            for item in self.negat_train_tfidf:
                sum_dot_product = 0
                for vec in item[0]:
                    if vec in items[0]:
                        acc_dot_product = item[0][vec] * items[0][vec]
                        if acc_dot_product == 0.0:
                            acc_dot_product = 0.001
                        sum_dot_product += acc_dot_product
                neg_dot.append(format(sum_dot_product, '.3f'))
            self.neg_dot_product.append(neg_dot)
        if debug:
            print '\nLength: ', len(self.neg_dot_product), '\n', self.neg_dot_product

    ############################### end of calculate dot product #####################################

    ############################### calculate magnitude ##############################################

    def calculate_magnitude(self):
        self.pos_train_norm = []
        self.neg_train_norm = []
        self.test_norm = []

        if debug:
            print '\nPositive training length normalization vector: '
        for items in self.posit_train_tfidf:
            posit_train_norm = math.sqrt(sum([math.pow(val,2) for val in items[0].values()]))
            self.pos_train_norm.append(format(posit_train_norm, '.3f'))
        if debug:
            print '\nLength: ', len(self.pos_train_norm), '\n', self.pos_train_norm

        if debug:
            print '\nNegative training length normalization vector: '
        for items in self.negat_train_tfidf:
            negat_train_norm = math.sqrt(sum([math.pow(val,2) for val in items[0].values()]))
            self.neg_train_norm.append(format(negat_train_norm, '.3f'))
        if debug:
            print '\nLength: ', len(self.neg_train_norm), '\n', self.neg_train_norm

        if debug:
            print '\nTesting length normalization vector: '
        for items in self.total_test:
            t_norm = math.sqrt(sum([math.pow(val,2) for val in items[0].values()]))
            self.test_norm.append(format(t_norm, '.3f'))
        if debug:
            print '\nLength: ', len(self.test_norm), '\n', self.test_norm

    ############################### end of calculate magnitude #########################################

    ############################### calculate cosine similarity ########################################

    def calculate_cosine_similarity(self):
        self.pos_cos_sim = []
        self.neg_cos_sim = []

        if debug:
            print '\nPositive cosine similarity: '
        for i in range(len(self.test_norm)):
            pos_sim = []
            for j in range(len(self.pos_train_norm)):
                if (float(self.test_norm[i]) * float(self.pos_train_norm[j])) == 0.000:
                    acc_sim = 0.000
                else:
                    acc_sim = float(self.pos_dot_product[i][j]) / (float(self.test_norm[i]) * \
                                                                float(self.pos_train_norm[j]))
                if acc_sim == 0.000:
                    acc_sim = 0.001
                pos_sim.append(format(acc_sim, '.3f'))
            self.pos_cos_sim.append(pos_sim)
        if debug:
            print '\nLength: ', len(self.pos_cos_sim), '\n', self.pos_cos_sim

        if debug:
            print '\nNegative cosine similarity: '
        for i in range(len(self.test_norm)):
            neg_sim = []
            for j in range(len(self.neg_train_norm)):
                if (float(self.test_norm[i]) * float(self.neg_train_norm[j])) == 0.000:
                    acc_sim = 0.000
                else:
                    acc_sim = float(self.neg_dot_product[i][j]) / (float(self.test_norm[i]) * \
                                                                float(self.neg_train_norm[j]))
                if acc_sim == 0.000:
                    acc_sim = 0.001
                neg_sim.append(format(acc_sim, '.3f'))
            self.neg_cos_sim.append(neg_sim)
        if debug:
            print '\nLength: ', len(self.neg_cos_sim), '\n', self.neg_cos_sim

    ############################### end of calculate cosine similarity #################################

    ############################### calculate product of cosine similarity #############################

    def calculate_product_cosines(self):
        self.sum_pos_cos_sim = []
        self.sum_neg_cos_sim = []

        if debug:
            print '\nSum of positive cosine similarity: '
        for i, items in enumerate(self.pos_cos_sim):
            pos_sum = 0
            for item in items:
                pos_sum += float(item)
            self.sum_pos_cos_sim.append(pos_sum)
        if debug:
            print '\nLength: ', len(self.sum_pos_cos_sim), '\n', self.sum_pos_cos_sim

        if debug:
            print '\nSum of negative cosine similarity: '
        for i, items in enumerate(self.neg_cos_sim):
            neg_sum = 0
            for item in items:
                neg_sum += float(item)
            self.sum_neg_cos_sim.append(neg_sum)
        if debug:
            print '\nLength: ', len(self.sum_neg_cos_sim), '\n', self.sum_neg_cos_sim

    ############################### end of calculate product of cosine similarity ######################

    ############################### calculate naive bayes probability ##################################
    def calculate_naive_bayes(self):
        self.prob_pos_train = format((len(self.posit_train_tfidf) / len(self.total_train_tfidf)), '.3f')
        self.prob_neg_train = format((len(self.negat_train_tfidf) / len(self.total_train_tfidf)), '.3f')

        print '\nProbability of positive training tweets: ', self.prob_pos_train
        print 'Probability of negative training tweets: ', self.prob_neg_train

        self.prob_pos = []
        self.prob_neg = []

        for items in self.sum_pos_cos_sim:
            self.prob_pos.append(float(items) * float(self.prob_pos_train))
        print '\nProbabilities of positive: ', len(self.prob_pos), '\n', self.prob_pos

        for items in self.sum_neg_cos_sim:
            self.prob_neg.append(float(items) * float(self.prob_neg_train))
        print '\nProbabilities of negative: ', len(self.prob_neg), '\n', self.prob_neg

    ############################### end of calculate naive bayes probability ###########################

    ############################### determine positive or negative #####################################
    def determine_pos_neg_tweets(self):
        print '\nDetermining..... : '
        self.classify_tweets = []
        count_pos, count_neg = 0, 0
        for i in range(len(self.prob_pos)):
            if self.prob_pos[i] >= self.prob_neg[i]:
                count_pos += 1
                self.classify_tweets.append("Positive")
            else:
                count_neg += 1
                self.classify_tweets.append("Negative")
        print '\nTest classified: ', len(self.classify_tweets), '\n', self.classify_tweets
        print 'Positve, Negative: ', count_pos, count_neg, '\n'

        ################################# precision & recall ###################
        actual_pos, actual_neg = 0, 0
        for i, items in enumerate(self.classify_tweets):
            if items == 'Positive':
                items = 0
                if items == int(self.total_test[i][1]):
                    actual_pos += 1
            else:
                actual_neg += 1
        print '\nActual Pos: ', actual_pos
        print '\nActual Neg: ', actual_neg

        precision_pos = actual_pos / count_pos
        recall_pos = actual_pos / len(self.total_test)
        print '\nPrecision, Recall (Pos): ', precision_pos, recall_pos

        precision_neg = actual_neg / count_neg
        recall_neg = actual_neg / len(self.total_test)
        print '\nPrecision, Recall (Neg): ', precision_neg, recall_neg
        ##################################################################################

        avg_precision = (precision_pos + precision_neg) / 2
        avg_recall = (recall_pos + recall_neg) / 2

        acc = (2 * (avg_precision * avg_recall)) / (avg_precision + avg_recall)
        print '\nAccuracy: ', acc * 100, '\n'

    ############################### end of determine positive or negative ##############################

a = Accuracy()

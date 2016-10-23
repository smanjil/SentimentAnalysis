
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
                                          if i < int(len(self.positive_tfidf) * 0.9)]
        self.negat_train_tfidf = [j for i, j in enumerate(self.negative_tfidf) \
                                          if i < int(len(self.negative_tfidf) * 0.9)]

        self.total_train_tfidf = self.posit_train_tfidf + self.negat_train_tfidf

        print '\nTotal positive training data: ', len(self.posit_train_tfidf)
        print 'Total negative training data: ', len(self.negat_train_tfidf)
        print 'Total training data: ', len(self.total_train_tfidf), '\n'


        self.pos_test = [j for i, j in enumerate(self.positive_tfidf) \
                                          if i >= int(len(self.positive_tfidf) * 0.9)]
        self.neg_test = [j for i, j in enumerate(self.negative_tfidf) \
                                          if i >= int(len(self.negative_tfidf) * 0.9)]

        print '\nTotal positive testing data: ', len(self.pos_test)
        print 'Total negative testing data: ', len(self.neg_test), '\n'

        self.calculate_dot_product()
        self.calculate_magnitude()
        self.calculate_cosine_similarity()
        self.calculate_product_cosines()
        self.calculate_naive_bayes()
        self.determine_pos_neg_tweets()

    ############################### calculate dot product #####################################

    def calculate_dot_product(self):
        self.pos_test_pos_train_dot_product = []
        self.pos_test_neg_train_dot_product = []
        self.neg_test_pos_train_dot_product = []
        self.neg_test_neg_train_dot_product = []

        # print '\nPositive Test Positive Train dot product: '
        for items in self.pos_test:
            pos_pos_dot_product = []
            for item in self.posit_train_tfidf:
                sum_dot_product = 0
                for vec in item[0]:
                    if vec in items[0]:
                        acc_dot_product = item[0][vec] * items[0][vec]
                        if acc_dot_product == 0.0:
                            acc_dot_product = 0.001
                        sum_dot_product += acc_dot_product
                pos_pos_dot_product.append(format(sum_dot_product, '.3f'))
            self.pos_test_pos_train_dot_product.append(pos_pos_dot_product)
        # print '\nLength: ', len(self.pos_test_pos_train_dot_product), '\n', self.pos_test_pos_train_dot_product

        # print '\nPositive Test Negative Train dot product: '
        for items in self.pos_test:
            pos_neg_dot_product = []
            for item in self.negat_train_tfidf:
                sum_dot_product = 0
                for vec in item[0]:
                    if vec in items[0]:
                        acc_dot_product = item[0][vec] * items[0][vec]
                        if acc_dot_product == 0.0:
                            acc_dot_product = 0.001
                        sum_dot_product += acc_dot_product
                pos_neg_dot_product.append(format(sum_dot_product, '.3f'))
            self.pos_test_neg_train_dot_product.append(pos_neg_dot_product)
        # print '\nLength: ', len(self.pos_test_neg_train_dot_product), '\n', self.pos_test_neg_train_dot_product

        # print '\nNegative Test Positive Train dot product: '
        for items in self.neg_test:
            neg_pos_dot_product = []
            for item in self.posit_train_tfidf:
                sum_dot_product = 0
                for vec in item[0]:
                    if vec in items[0]:
                        acc_dot_product = item[0][vec] * items[0][vec]
                        if acc_dot_product == 0.0:
                            acc_dot_product = 0.001
                        sum_dot_product += acc_dot_product
                neg_pos_dot_product.append(format(sum_dot_product, '.3f'))
            self.neg_test_pos_train_dot_product.append(neg_pos_dot_product)
        # print '\nLength: ', len(self.neg_test_pos_train_dot_product), '\n', self.neg_test_pos_train_dot_product

        # print '\nNegative Test Negative Train dot product: '
        for items in self.neg_test:
            neg_neg_dot_product = []
            for item in self.negat_train_tfidf:
                sum_dot_product = 0
                for vec in item[0]:
                    if vec in items[0]:
                        acc_dot_product = item[0][vec] * items[0][vec]
                        if acc_dot_product == 0.0:
                            acc_dot_product = 0.001
                        sum_dot_product += acc_dot_product
                neg_neg_dot_product.append(format(sum_dot_product, '.3f'))
            self.neg_test_neg_train_dot_product.append(neg_neg_dot_product)
        # print '\nLength: ', len(self.neg_test_neg_train_dot_product), '\n', self.neg_test_neg_train_dot_product

    ############################### end of calculate dot product #####################################

    ############################### calculate magnitude ##############################################

    def calculate_magnitude(self):
        self.pos_train_norm = []
        self.neg_train_norm = []
        self.pos_test_norm = []
        self.neg_test_norm = []

        # print '\nPositive training length normalization vector: '
        for items in self.posit_train_tfidf:
            posit_train_norm = math.sqrt(sum([math.pow(val,2) for val in items[0].values()]))
            self.pos_train_norm.append(format(posit_train_norm, '.3f'))
        # print '\nLength: ', len(self.pos_train_norm), '\n', self.pos_train_norm

        # print '\nNegative training length normalization vector: '
        for items in self.negat_train_tfidf:
            negat_train_norm = math.sqrt(sum([math.pow(val,2) for val in items[0].values()]))
            self.neg_train_norm.append(format(negat_train_norm, '.3f'))
        # print '\nLength: ', len(self.neg_train_norm), '\n', self.neg_train_norm

        # print '\nPositive testing length normalization vector: '
        for items in self.pos_test:
            pos_test_norm = math.sqrt(sum([math.pow(val,2) for val in items[0].values()]))
            self.pos_test_norm.append(format(pos_test_norm, '.3f'))
        # print '\nLength: ', len(self.pos_test_norm), '\n', self.pos_test_norm

        # print '\nNegative testing length normalization vector: '
        for items in self.neg_test:
            neg_test_norm = math.sqrt(sum([math.pow(val,2) for val in items[0].values()]))
            self.neg_test_norm.append(format(neg_test_norm, '.3f'))
        # print '\nLength: ', len(self.neg_test_norm), '\n', self.neg_test_norm

    ############################### end of calculate magnitude #########################################

    ############################### calculate cosine similarity ########################################

    def calculate_cosine_similarity(self):
        self.pos_pos_cos_sim = []
        self.pos_neg_cos_sim = []
        self.neg_pos_cos_sim = []
        self.neg_neg_cos_sim = []

        # print '\nPositive testing, positive training cosine similarity: '
        for i in range(len(self.pos_test_norm)):
            pos_pos_sim = []
            for j in range(len(self.pos_train_norm)):
                if (float(self.pos_test_norm[i]) * float(self.pos_train_norm[j])) == 0.000:
                    acc_sim = 0.000
                else:
                    acc_sim = float(self.pos_test_pos_train_dot_product[i][j]) / (float(self.pos_test_norm[i]) * \
                                                                float(self.pos_train_norm[j]))
                if acc_sim == 0.000:
                    acc_sim = 0.001
                pos_pos_sim.append(format(acc_sim, '.3f'))
            self.pos_pos_cos_sim.append(pos_pos_sim)
        # print '\nLength: ', len(self.pos_pos_cos_sim), '\n', self.pos_pos_cos_sim

        # print '\nPositive testing, negative training cosine similarity: '
        for i in range(len(self.pos_test_norm)):
            pos_neg_sim = []
            for j in range(len(self.neg_train_norm)):
                if (float(self.pos_test_norm[i]) * float(self.neg_train_norm[j])) == 0.000:
                    acc_sim = 0.000
                else:
                    acc_sim = float(self.pos_test_neg_train_dot_product[i][j]) / (float(self.pos_test_norm[i]) * \
                                                                float(self.neg_train_norm[j]))
                if acc_sim == 0.000:
                    acc_sim = 0.001
                pos_neg_sim.append(format(acc_sim, '.3f'))
            self.pos_neg_cos_sim.append(pos_neg_sim)
        # print '\nLength: ', len(self.pos_neg_cos_sim), '\n', self.pos_neg_cos_sim

        # print '\nNegative testing, positive training cosine similarity: '
        for i in range(len(self.neg_test_norm)):
            neg_pos_sim = []
            for j in range(len(self.pos_train_norm)):
                if (float(self.neg_test_norm[i]) * float(self.pos_train_norm[j])) == 0.000:
                    acc_sim = 0.000
                else:
                    acc_sim = float(self.neg_test_pos_train_dot_product[i][j]) / (float(self.neg_test_norm[i]) * \
                                                                float(self.pos_train_norm[j]))
                if acc_sim == 0.000:
                    acc_sim = 0.001
                neg_pos_sim.append(format(acc_sim, '.3f'))
            self.neg_pos_cos_sim.append(neg_pos_sim)
        # print '\nLength: ', len(self.neg_pos_cos_sim), '\n', self.neg_pos_cos_sim

        # print '\nNegative testing, negative training cosine similarity: '
        for i in range(len(self.neg_test_norm)):
            neg_neg_sim = []
            for j in range(len(self.neg_train_norm)):
                if (float(self.neg_test_norm[i]) * float(self.neg_train_norm[j])) == 0.000:
                    acc_sim = 0.000
                else:
                    acc_sim = float(self.neg_test_neg_train_dot_product[i][j]) / (float(self.neg_test_norm[i]) * \
                                                                float(self.neg_train_norm[j]))
                if acc_sim == 0.000:
                    acc_sim = 0.001
                neg_neg_sim.append(format(acc_sim, '.3f'))
            self.neg_neg_cos_sim.append(neg_neg_sim)
        # print '\nLength: ', len(self.neg_neg_cos_sim), '\n', self.neg_neg_cos_sim

    ############################### end of calculate cosine similarity #################################

    ############################### calculate product of cosine similarity #############################

    def calculate_product_cosines(self):
        self.product_pos_pos_cos_sim = []
        self.product_pos_neg_cos_sim = []
        self.product_neg_pos_cos_sim = []
        self.product_neg_neg_cos_sim = []

        # print '\nProduct of positive testing, positive training cosine similarity: '
        for i, items in enumerate(self.pos_pos_cos_sim):
            pos_pos_pro = 1
            for item in items:
                pos_pos_pro *= float(item)
            self.product_pos_pos_cos_sim.append(pos_pos_pro)
        # print '\nLength: ', len(self.product_pos_pos_cos_sim), '\n', self.product_pos_pos_cos_sim

        # print '\nProduct of positive testing, negative training cosine similarity: '
        for i, items in enumerate(self.pos_neg_cos_sim):
            pos_neg_pro = 1
            for item in items:
                pos_neg_pro *= float(item)
            self.product_pos_neg_cos_sim.append(pos_neg_pro)
        # print '\nLength: ', len(self.product_pos_neg_cos_sim), '\n', self.product_pos_neg_cos_sim

        # print '\nProduct of negative testing, positive training cosine similarity: '
        for i, items in enumerate(self.neg_pos_cos_sim):
            neg_pos_pro = 1
            for item in items:
                neg_pos_pro *= float(item)
            self.product_neg_pos_cos_sim.append(neg_pos_pro)
        # print '\nLength: ', len(self.product_neg_pos_cos_sim), '\n', self.product_neg_pos_cos_sim

        # print '\nProduct of negative testing, negative training cosine similarity: '
        for i, items in enumerate(self.neg_neg_cos_sim):
            neg_neg_pro = 1
            for item in items:
                neg_neg_pro *= float(item)
            self.product_neg_neg_cos_sim.append(neg_neg_pro)
        # print '\nLength: ', len(self.product_neg_neg_cos_sim), '\n', self.product_neg_neg_cos_sim

    ############################### end of calculate product of cosine similarity ######################

    ############################### calculate naive bayes probability ##################################
    def calculate_naive_bayes(self):
        self.prob_pos_train = format((len(self.posit_train_tfidf) / len(self.total_train_tfidf)), '.3f')
        self.prob_neg_train = format((len(self.negat_train_tfidf) / len(self.total_train_tfidf)), '.3f')

        # print '\nProbability of positive training tweets: ', self.prob_pos_train
        # print 'Probability of negative training tweets: ', self.prob_neg_train

        self.prob_pos_pos = []
        self.prob_pos_neg = []
        self.prob_neg_pos = []
        self.prob_neg_neg = []

        # print '\nProbabilities of positive testing and positive training: '
        for items in self.product_pos_pos_cos_sim:
            self.prob_pos_pos.append(float(items) * float(self.prob_pos_train))
        # print '\nProbabilities of pos pos: ', len(self.prob_pos_pos), '\n', self.prob_pos_pos

        # print '\nProbabilities of positive testing and negative training: '
        for items in self.product_pos_neg_cos_sim:
            self.prob_pos_neg.append(float(items) * float(self.prob_neg_train))
        # print '\nProbabilities of pos neg: ', len(self.prob_pos_neg), '\n', self.prob_pos_neg

        # print '\nProbabilities of negative testing and positive training: '
        for items in self.product_neg_pos_cos_sim:
            self.prob_neg_pos.append(float(items) * float(self.prob_pos_train))
        # print '\nProbabilities of neg pos: ', len(self.prob_neg_pos), '\n', self.prob_neg_pos

        # print '\nProbabilities of negative testing and negative training: '
        for items in self.product_neg_neg_cos_sim:
            self.prob_neg_neg.append(float(items) * float(self.prob_neg_train))
        # print '\nProbabilities of neg neg: ', len(self.prob_neg_neg), '\n', self.prob_neg_neg

    ############################### end of calculate naive bayes probability ###########################

    ############################### determine positive or negative #####################################
    def determine_pos_neg_tweets(self):
        print '\nFor positive testing: '
        self.classify_tweets = []
        count_pos, count_neg, count_neu = 0, 0, 0
        for i in range(len(self.prob_pos_pos)):
            if self.prob_pos_pos[i] > self.prob_pos_neg[i]:
                count_pos += 1
                self.classify_tweets.append("Positive")
            elif self.prob_pos_pos[i] < self.prob_pos_neg[i]:
                count_neg += 1
                self.classify_tweets.append("Negative")
            else:
                count_neu += 1
                self.classify_tweets.append("Neutral")
        print '\nPositive test classified: ', len(self.classify_tweets), '\n', self.classify_tweets
        print 'Positve, Negative, Neutral: ', count_pos, count_neg, count_neu, '\n'

        ################################# precision & recall positive ###################
        actual_pos = 0
        for i, items in enumerate(self.classify_tweets):
            if items == 'Positive':
                items = 0
                if items == int(self.pos_test[i][1]):
                    actual_pos += 1
        print '\nActual Pos: ', actual_pos, '\n'

        precision_pos = actual_pos / count_pos
        recall_pos = actual_pos / len(self.pos_test)
        ##################################################################################

        print '\nFor negative testing: '
        self.classify_tweets = []
        count_pos, count_neg, count_neu = 0, 0, 0
        for i in range(len(self.prob_neg_pos)):
            if self.prob_neg_pos[i] > self.prob_neg_neg[i]:
                count_pos += 1
                self.classify_tweets.append("Positive")
            elif self.prob_neg_pos[i] < self.prob_neg_neg[i]:
                count_neg += 1
                self.classify_tweets.append("Negative")
            else:
                count_neu += 1
                self.classify_tweets.append("Neutral")
        print '\nNegative test classified: ', len(self.classify_tweets), '\n', self.classify_tweets
        print 'Positve, Negative, Neutral: ', count_pos, count_neg, count_neu, '\n'

        ################################# precision & recall negative ###################
        actual_neg = 0
        for i, items in enumerate(self.classify_tweets):
            if items == 'Negative':
                items = 4
                if items == int(self.neg_test[i][1]):
                    actual_neg += 1
        print '\nActual Neg: ', actual_neg, '\n'

        precision_neg = actual_neg / count_neg
        recall_neg = actual_neg / len(self.neg_test)
        ##################################################################################

        ################################## accuracy ####################################
        avg_precision = (precision_pos + precision_neg) / 2
        avg_recall = (recall_pos + recall_neg) / 2

        acc = (2 * (avg_precision * avg_recall)) / (avg_precision + avg_recall)
        print '\nAccuracy: ', acc * 100, '\n'
        ################################################################################

    ############################### end of determine positive or negative ##############################

a = Accuracy()

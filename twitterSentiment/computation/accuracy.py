

from __future__ import division
import math
from config import debug
from naive_bayes import NaiveBayes

class Accuracy:
    def __init__(self):
        nb = NaiveBayes()

        self.positive_tfidf = nb.positive_train_tfidf
        self.negative_tfidf = nb.negative_train_tfidf

        self.tr = 0.8
        self.te = 1 - self.tr
        self.tot = len(self.positive_tfidf + self.negative_tfidf)

        self.posit_train_tfidf = [j for i, j in enumerate(self.positive_tfidf) \
                                          if i < int(len(self.positive_tfidf) * self.tr)]
        self.negat_train_tfidf = [j for i, j in enumerate(self.negative_tfidf) \
                                          if i < int(len(self.negative_tfidf) * self.tr)]
        self.total_train_tfidf = self.posit_train_tfidf + self.negat_train_tfidf

        self.pos_test = [j for i, j in enumerate(self.positive_tfidf) \
                                          if i >= int(len(self.positive_tfidf) * self.tr)]
        self.neg_test = [j for i, j in enumerate(self.negative_tfidf) \
                                          if i >= int(len(self.negative_tfidf) * self.tr)]
        self.total_test = self.pos_test + self.neg_test

        ###### file open #########
        self.fo = open('data_file/{0}-{1}-{2}.txt' .format(self.tr, self.te, self.tot), 'w+')

        ###### file write #########
        self.fo.write('\nTotal positive, negative data: {0} {1}' .format(len(self.positive_tfidf), len(self.negative_tfidf)))

        ###### file write #########
        self.fo.write('\n\nTotal positive, negative data, total training: {0} {1} {2}' \
                                .format(len(self.posit_train_tfidf), len(self.negat_train_tfidf), len(self.total_train_tfidf)))

        ###### file write #########
        self.fo.write('\n\nTotal positive, negative data, total testing: {0} {1} {2}' \
                 .format(len(self.pos_test), len(self.neg_test), len(self.total_test)))

        print '\nAccuracy (accuracy.py): ', '\n'

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

        self.pos_train_norm = [format(math.sqrt(sum([math.pow(val,2) for val in items[0].values()])), '.3f') \
                                    for items in self.posit_train_tfidf]

        self.neg_train_norm = [format(math.sqrt(sum([math.pow(val,2) for val in items[0].values()])), '.3f') \
                                    for items in self.negat_train_tfidf]

        self.test_norm = [format(math.sqrt(sum([math.pow(val,2) for val in items[0].values()])), '.3f') \
                                    for items in self.total_test]

    ############################### end of calculate magnitude #########################################

    ############################### calculate cosine similarity ########################################

    def calculate_cosine_similarity(self):
        self.pos_cos_sim = []
        self.neg_cos_sim = []

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

    ############################### end of calculate cosine similarity #################################

    ############################### calculate product of cosine similarity #############################

    def calculate_product_cosines(self):
        self.product_pos_cos_sim = []
        self.product_neg_cos_sim = []

        for i, items in enumerate(self.pos_cos_sim):
            pos_pro = 1
            for item in items:
                pos_pro *= float(item)
            self.product_pos_cos_sim.append(pos_pro)

        for i, items in enumerate(self.neg_cos_sim):
            neg_pro = 1
            for item in items:
                neg_pro *= float(item)
            self.product_neg_cos_sim.append(neg_pro)    

    ############################### end of calculate product of cosine similarity ######################

    ############################### calculate naive bayes probability ##################################
    def calculate_naive_bayes(self):
        self.prob_pos_train = format((len(self.posit_train_tfidf) / len(self.total_train_tfidf)), '.3f')
        self.prob_neg_train = format((len(self.negat_train_tfidf) / len(self.total_train_tfidf)), '.3f')

        self.prob_pos = (float(items) * float(self.prob_pos_train) for items in self.product_pos_cos_sim)
        self.prob_neg = (float(items) * float(self.prob_neg_train) for items in self.product_neg_cos_sim)

    ############################### end of calculate naive bayes probability ###########################

    ############################### determine positive or negative #####################################
    def determine_pos_neg_tweets(self):
        print '\nDetermining..... : '
        self.classify_tweets = []
        count_pos, count_neg = 0, 0
        for x, y in zip(self.prob_pos, self.prob_neg):
            if x >= y:
                count_pos += 1
                self.classify_tweets.append("Positive")
            else:
                count_neg += 1
                self.classify_tweets.append("Negative")
        print 'Positve, Negative: ', count_pos, count_neg, '\n'

        ########## file write ###################
        self.fo.write('\n\nTest classified: {0}\n{1}' .format(len(self.classify_tweets), self.classify_tweets))
        self.fo.write('\n\nPositive, Negative: {0} {1}' .format(count_pos, count_neg))

        ################################# precision & recall ###################
        actual_pos, actual_neg = 0, 0

        pc = self.classify_tweets[:len(self.pos_test)]
        print len(pc)

        nc = self.classify_tweets[len(self.pos_test):]
        print len(nc)

        ############ file write ###########
        self.fo.write('\n\nClassified positive, negative: {0} {1}' .format(len(pc), len(nc)))

        for i, items in enumerate(pc):
            if items == 'Positive':
                items = 0
                if items == int(self.pos_test[i][1]):
                    actual_pos += 1

        for i, items in enumerate(nc):
            if items == 'Negative':
                items = 4
                if items == int(self.neg_test[i][1]):
                    actual_neg += 1

        print '\nActual Pos: ', actual_pos
        print 'Actual Neg: ', actual_neg

        ############ file write ###########
        self.fo.write('\n\nActual positive, negative: {0} {1}'.format(actual_pos, actual_neg))

        try:
            ##### contingency(confusion) matrix ##########
            tp = actual_pos
            fn = len(self.pos_test) - tp
            tn = actual_neg
            fp = len(self.neg_test) - tn

            print '\nTP, FN, TN, FP: ', tp, fn, tn, fp, '\n'

            ############ file write ###########
            self.fo.write('\n\nTP, FN, TN, FP: {0} {1} {2} {3}' .format(tp, fn, tn, fp))
            ####### end ####################

            #################### precision, recall and accuracy ######################

            print 'TP + FP: ', tp + fp
            print 'TP + FN: ', tp + fn
            print 'TN + FN: ', tn + fn
            print 'TN + FP: ', tn + fp

            pos_precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
            pos_recall = 1.0 if tp + fn == 0 else tp / (tp + fn)    
            neg_precision = 1.0 if tn + fn == 0 else tn / (tn + fn)
            neg_recall = 1.0 if tn + fp == 0 else tn / (tn + fp)

            avg_precision = (pos_precision + neg_precision) / 2
            avg_recall = (pos_recall + neg_recall) / 2

            f_measure = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)
            # accuracy = (pos_f_measure + neg_f_measure) / 2
            print '\nF-measure: ', f_measure * 100, ' percent....'

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            print '\nAccuracy: ', accuracy * 100, ' percent....'

            ############ file write ###########
            self.fo.write('\n\nAvg. Precision, Recall: {0} {1}' .format(avg_precision, avg_recall))
            # self.fo.write('\n\nPos Neg F-Measure: {0} {1}' .format(pos_f_measure, neg_f_measure))

            ############ file write ###########
            self.fo.write('\n\nF-measure: {0} percent....' .format(f_measure * 100))
            self.fo.write('\n\nAccuracy: {0} percent....' .format(accuracy * 100))

        except ZeroDivisionError:
            print '\nDivide by zero occured!'

            ############ file write ###########33
            self.fo.write('\n\nDivision by zero occurred!!!!')

    ############################### end of determine positive or negative ##############################

a = Accuracy()

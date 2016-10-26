
from config import debug
import math
from training_tf_idf import TrainingTfIdf
from testing_tf_idf import TestingTfIdf

class CosineSimilarity:
    # constructor (imports from training_tf_idf and testing_tf_idf)
    def __init__(self, trnum, tenum):

        self.trnum, self.tenum = trnum, tenum

        self.train_tf_idf = TrainingTfIdf(self.trnum, self.tenum).tot_tf_idf
        self.test_tf_idf = TestingTfIdf(self.trnum, self.tenum).tot_tf_idf

        if debug:
            print '\nCosine Similarity: (cosine_similarity.py)\n'

        self.separate_positive_and_negative_vectors()
        self.calculate_dot_product()
        self.calculate_magnitude()
        self.calculate_cosine_similarity()
        self.calculate_product_cosines()

    # separate positive and negative vectors
    def separate_positive_and_negative_vectors(self):
        self.positive_train_tfidf = []
        self.negative_train_tfidf = []

        for items in self.train_tf_idf:
            if items[1] == '0':
                self.positive_train_tfidf.append(items)
            else:
                self.negative_train_tfidf.append(items)
        return (self.positive_train_tfidf, self.negative_train_tfidf)

    # calculate dot product
    def calculate_dot_product(self):
        self.total_pos_dot_product = []
        self.total_neg_dot_product = []
        
        test_val = []
        for items in self.test_tf_idf:
            inner_test = []
            for item in items:
                inner_test.append(items[item])
            test_val.append(inner_test)
        if debug:
            print test_val, '\n'

        pos_val = []
        for items in self.positive_train_tfidf:
            inner_pos = []
            for item in items[0]:
                inner_pos.append(items[0][item])
            pos_val.append(inner_pos)
        if debug:
            print pos_val, '\n'

        neg_val = []
        for items in self.negative_train_tfidf:
            inner_neg = []
            for item in items[0]:
                inner_neg.append(items[0][item])
            neg_val.append(inner_neg)
        if debug:
            print neg_val, '\n'

        for i, items in enumerate(test_val):
            pos_dot_product = []
            for item in pos_val:
                sum_dot_product = sum([x * y if x * y != 0.0 else 0.001 for x, y in zip(items, item)])
                pos_dot_product.append(format(sum_dot_product, '.3f'))
            self.total_pos_dot_product.append(pos_dot_product)
        if debug:
            print self.total_pos_dot_product                

        for i, items in enumerate(test_val):
            neg_dot_product = []
            for item in neg_val:
                sum_dot_product = sum([x * y if x * y != 0.0 else 0.001 for x, y in zip(items, item)])
                neg_dot_product.append(format(sum_dot_product, '.3f'))
            self.total_neg_dot_product.append(neg_dot_product)
        if debug:
            print self.total_neg_dot_product
        
    # length normalization of vectors
    def calculate_magnitude(self):
        self.test_normalization = []
        self.pos_train_normalization = []
        self.neg_train_normalization = []

        # length normalization for test vectors
        self.test_normalization = [format(math.sqrt(sum([math.pow(val,2) for val in items.values()])), '.3f') \
                                        for items in self.test_tf_idf]
        if debug:
            print self.test_normalization, '\n'

        # length normalization for positive training vectors
        self.pos_train_normalization = [format( math.sqrt(sum([math.pow(val,2) for val in items[0].values()])) , '.3f') \
                                            for items in self.positive_train_tfidf]
        if debug:
            print self.pos_train_normalization, '\n'

        # length normalization for negative training vectors
        self.neg_train_normalization = [format(math.sqrt(sum([math.pow(val,2) for val in items[0].values()])) , '.3f') \
                                            for items in self.negative_train_tfidf]
        if debug:
            print self.neg_train_normalization, '\n'

    # calculate cosine similarity
    def calculate_cosine_similarity(self):
        self.pos_cos_sim = []
        self.neg_cos_sim = []

        # positive cosine similarities
        for i, items in enumerate(self.test_normalization):
            pos_sim = []
            for j, jtems in enumerate(self.pos_train_normalization):
                if (float(items) * float(jtems)) == 0.000:
                    sim = 0.000
                else:
                    sim = float(self.total_pos_dot_product[i][j]) / (float(items) * float(jtems))
                if sim == 0.000:
                    sim = 0.001
                pos_sim.append(format(sim, '.3f'))
            self.pos_cos_sim.append(pos_sim)
        if debug:
            print self.pos_cos_sim

        # negative cosine similarities
        for i, items in enumerate(self.test_normalization):
            neg_sim = []
            for j, jtems in enumerate(self.neg_train_normalization):
                if (float(items) * float(jtems)) == 0.000:
                    sim = 0.000
                else:
                    sim = float(self.total_neg_dot_product[i][j]) / (float(items) * float(jtems))
                if sim == 0.000:
                    sim = 0.001
                neg_sim.append(format(sim, '.3f'))
            self.neg_cos_sim.append(neg_sim)
        if debug:
            print self.neg_cos_sim 

    # calculate product of positive and negative cosine similarities
    def calculate_product_cosines(self):
        self.product_pos_cos_sim = []
        self.product_neg_cos_sim = []

        # product of positive cosine similarities
        for i, items in enumerate(self.pos_cos_sim):
            pos_pro = 1
            for item in items:
                pos_pro *= float(item)
            self.product_pos_cos_sim.append(pos_pro)

        # product of negative cosine similarities
        for i, items in enumerate(self.neg_cos_sim):
            neg_pro = 1
            for item in items:
                neg_pro *= float(item)
            self.product_neg_cos_sim.append(neg_pro)
        if debug:
            print self.product_pos_cos_sim, '\n', self.product_neg_cos_sim
        return (self.product_pos_cos_sim, self.product_neg_cos_sim)

# cs = CosineSimilarity()

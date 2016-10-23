
from config import debug
import math
from training_tf_idf import TrainingTfIdf
from testing_tf_idf import TestingTfIdf

class CosineSimilarity:
    # constructor (imports from training_tf_idf and testing_tf_idf)
    def __init__(self):
        training = TrainingTfIdf()
        testing = TestingTfIdf()

        training.calculate_tf_idf()
        testing.calculate_tf_idf()

        self.train_tf_idf = training.tot_tf_idf
        self.test_tf_idf = testing.tot_tf_idf
        
        # debug program
        if debug:
            print 'Testing: \n', self.test_tf_idf
            print '\nTraining tf-idf: \n', self.train_tf_idf, '\n'
            print 'Testing tf-idf: \n', self.test_tf_idf[0], '\n'

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
        
        # debug program
        if debug:
            print '\nPositve Training Data: ', self.positive_train_tfidf, '\n'
            print '\nNegatve Training Data: ', self.negative_train_tfidf, '\n'
            print '\nTotal Positve Training Data: ', len(self.positive_train_tfidf), '\n'
            print 'Total Negative Training Data: ', len(self.negative_train_tfidf), '\n'
            print 'Total Testing Data: ', len(self.test_tf_idf), '\n'

    # calculate dot product
    def calculate_dot_product(self):
        self.total_pos_dot_product = []
        self.total_neg_dot_product = []

        # dot products for positive training vectors and testing vectors        
        # debug program
        if debug:
            print '\nPositive dot product: '
        for items in self.test_tf_idf:
            pos_dot_product = []
            # debug program
            if debug:
                print '\nItems Start-----------'
                print items, '\n'
            for item in self.positive_train_tfidf:
                sum_dot_product = 0
                # debug program
                if debug:
                    print '\nItem[0] Start-----------'
                    print item[0]
                for vec in item[0]:
                    if vec in items:
                        dot_product = item[0][vec] * items[vec]
                        if dot_product == 0.0:
                            dot_product = 0.001
                        sum_dot_product += dot_product
                pos_dot_product.append(format(sum_dot_product, '.3f'))
            self.total_pos_dot_product.append(pos_dot_product)
        
        # debug program
        if debug:
            print '\n', self.total_pos_dot_product

        # dot products for negative training vectors and testing vectors
        # debug program
        if debug:
            print '\nNegative dot product: '
        for items in self.test_tf_idf:
            neg_dot_product = []
            # debug program
            if debug:
                print '\nItems Start-----------'
                print items, '\n'
            for item in self.negative_train_tfidf:
                sum_dot_product = 0
                # debug program
                if debug:
                    print '\nItem[0] Start-----------'
                    print item[0]
                for vec in item[0]:
                    if vec in items:
                        dot_product = item[0][vec] * items[vec]
                        if dot_product == 0.0:
                            dot_product = 0.001
                        sum_dot_product += dot_product
                neg_dot_product.append(format(sum_dot_product, '.3f'))
            self.total_neg_dot_product.append(neg_dot_product)
        # debug program
        if debug:
            print '\n', self.total_neg_dot_product, '\n'

    # length normalization of vectors
    def calculate_magnitude(self):
        self.test_normalization = []
        self.pos_train_normalization = []
        self.neg_train_normalization = []

        # length normalization for test vectors
        # debug program
        if debug:
            print 'Testing length normalization vector: '
        for items in self.test_tf_idf:
            test_norm = math.sqrt(sum([math.pow(val,2) for val in items.values()]))
            self.test_normalization.append(format(test_norm, '.3f'))
        # debug program
        if debug:
            print self.test_normalization, '\n'

        # length normalization for positive training vectors
        # debug program
        if debug:
            print 'Positive training length normalization vector: '
        for items in self.positive_train_tfidf:
            pos_train_norm = math.sqrt(sum([math.pow(val,2) for val in items[0].values()]))
            self.pos_train_normalization.append(format(pos_train_norm, '.3f'))
        # debug program
        if debug:
            print self.pos_train_normalization, '\n'

        # length normalization for negative training vectors
        # debug program
        if debug:
            print 'Negative training length normalization vector: '
        for items in self.negative_train_tfidf:
            neg_train_norm = math.sqrt(sum([math.pow(val,2) for val in items[0].values()]))
            self.neg_train_normalization.append(format(neg_train_norm, '.3f'))
        # debug program
        if debug:
            print self.neg_train_normalization, '\n'

    # calculate cosine similarity
    def calculate_cosine_similarity(self):
        self.pos_cos_sim = []
        self.neg_cos_sim = []

        # positive cosine similarities
        # debug program
        if debug:
            print 'Positive Cosine Similarities: '
        for i in range(len(self.test_normalization)):
            pos_sim = []
            for j in range(len(self.pos_train_normalization)):
                # debug program
                if debug:
                    print type(float(self.total_pos_dot_product[i][j]))
                    print type(float(self.test_normalization[i]))
                    print type(float(self.pos_train_normalization[j]))
                if float(self.pos_train_normalization[j]) == 0.000:
                    sim = 0.000
                else:
                    sim = float(self.total_pos_dot_product[i][j]) / (float(self.test_normalization[i]) * \
                                                                float(self.pos_train_normalization[j]))
                if sim == 0.000:
                    sim = 0.001
                pos_sim.append(format(sim, '.3f'))
            self.pos_cos_sim.append(pos_sim)
        # debug program
        if debug:
            print self.pos_cos_sim, '\n'

        # negative cosine similarities
        # debug program
        if debug:
            print 'Negative Cosine Similarities: '
        for i in range(len(self.test_normalization)):
            neg_sim = []
            for j in range(len(self.neg_train_normalization)):
                if float(self.neg_train_normalization[j]) == 0.000:
                    sim = 0.000
                else:
                    sim = float(self.total_neg_dot_product[i][j]) / (float(self.test_normalization[i]) * \
                                                                float(self.neg_train_normalization[j]))
                if sim == 0.000:
                    sim = 0.001
                neg_sim.append(format(sim, '.3f'))
            self.neg_cos_sim.append(neg_sim)
        # debug program
        if debug:
            print self.neg_cos_sim, '\n'

    # calculate product of positive and negative cosine similarities
    def calculate_product_cosines(self):
        self.product_pos_cos_sim = []
        self.product_neg_cos_sim = []

        # product of positive cosine similarities
        # debug program
        if debug:
            print 'Product of positive cosine similarities: '
        for i, items in enumerate(self.pos_cos_sim):
            pos_pro = 1
            for item in items:
                pos_pro *= float(item)
            self.product_pos_cos_sim.append(pos_pro)
            # self.product_pos_cos_sim.append(format(pos_pro, '.8f'))
        # debug program
        if debug:
            print self.product_pos_cos_sim, '\n'

        # product of negative cosine similarities
        # debug program
        if debug:
            print 'Product of negative cosine similarities: '
        for i, items in enumerate(self.neg_cos_sim):
            neg_pro = 1
            for item in items:
                neg_pro *= float(item)
            self.product_neg_cos_sim.append(neg_pro)
            # self.product_neg_cos_sim.append(format(neg_pro, '.8f'))
        # debug program
        if debug:
            print self.product_neg_cos_sim, '\n'

# cs = CosineSimilarity()

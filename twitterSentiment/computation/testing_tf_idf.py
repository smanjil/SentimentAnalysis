
from config import debug
import math
from data import Data
from sdict import AlphaSortedDict

class TestingTfIdf:
    # constructor (imports from 'data.py')
    def __init__(self, trnum, tenum):

        self.trnum, self.tenum = trnum, tenum

        d = Data(self.trnum, self.tenum)
        self.test = d.sentence_filtered
        self.total_bag_words = d.total_bag_words

        if debug:
            print '\nTesting TF-IDF: (testing_tf_idf.py)\n'

        self.calculate_tf()
        self.calculate_idf()
        self.calculate_tf_idf()

    # calculate term frequency of testing data
    def calculate_tf(self):
        self.total_term_frequency = []
        self.document_frequency = AlphaSortedDict({})

        f = []
        for i in range(len(self.test)):
            twit = self.test[i].split()
            tf = []
            for item in self.total_bag_words:
                tf.append(twit.count(item))
            f.append(tf)
        if debug:
            print '\nf: ', f

        self.total_term_frequency = [AlphaSortedDict(zip(self.total_bag_words, items)) \
                                        for i, items in enumerate(f)]
        if debug:
            print '\ntotal_term_frequency: \n', self.total_term_frequency

        # calculate document frequency of testing data
        for items in self.total_term_frequency:
            for item in items:
                if item not in self.document_frequency:
                    self.document_frequency[item] = items[item]
                else:
                    self.document_frequency[item] += items[item]
        if debug:
            print '\ndocument_frequency: ', self.document_frequency

    # calculate inverse document frequency of testing data
    def calculate_idf(self):
        self.inverse_document_frequency = {items: (format(math.log(len(self.test) / self.document_frequency[items], 10), '.3f') \
                                                if self.document_frequency[items] != 0 else format(0, '.3f')) \
                                                    for items in self.document_frequency}
        self.inverse_document_frequency = AlphaSortedDict(self.inverse_document_frequency)
        if debug:
            print '\nInverse Document Frequency: ', self.inverse_document_frequency

    # calculate tf * idf of testing data
    def calculate_tf_idf(self):
        self.tot_tf_idf = []
        for items in self.total_term_frequency:
            sep_tf_idf = {}
            sep_tf_idf = {item: float(self.inverse_document_frequency[item]) * items[item] for item in items}
            sep_tf_idf = AlphaSortedDict(sep_tf_idf)
            self.tot_tf_idf.append(sep_tf_idf)
        if debug:
            print '\nTF-IDF: ', self.tot_tf_idf, '\n\n'
        return self.tot_tf_idf

# tf = TestingTfIdf()

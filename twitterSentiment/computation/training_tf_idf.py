
from config import debug
import math
from data import Data
from sdict import AlphaSortedDict

class TrainingTfIdf:
    # constructor (imports from 'data.py')
    def __init__(self, trnum, tenum):

        self.trnum, self.tenum = trnum, tenum

        d = Data(self.trnum, self.tenum)
        self.tweets = d.tweets
        self.total_bag_words = d.total_bag_words
        
        if debug:
            print '\nTraining TF-IDF: (training_tf_idf.py)\n'

        self.calculate_tf()
        self.calculate_idf()
        self.calculate_tf_idf()

    # calculate term frequency of training data
    def calculate_tf(self):
        self.total_term_frequency = []
        self.document_frequency = AlphaSortedDict({})

        f = []
        for i in range(len(self.tweets)):
            twit = self.tweets[i][1].split()
            tf = []
            for item in self.total_bag_words:
                tf.append(twit.count(item))
            f.append(tf)
        if debug:
            print '\nf: ', f

        self.total_term_frequency = [[AlphaSortedDict(zip(self.total_bag_words, items)), self.tweets[i][0]] \
                                        for i, items in enumerate(f)]
        if debug:
            print '\ntotal_term_frequency: \n', self.total_term_frequency

        # calculate document frequency
        for items in self.total_term_frequency:
            for item in items[0]:
                if item not in self.document_frequency:
                    self.document_frequency[item] = items[0][item]
                else:
                    self.document_frequency[item] += items[0][item] 
        if debug:
            print '\ndocument_frequency: ', self.document_frequency
        
    # calculate inverse document frequencysss of training data
    def calculate_idf(self):
        self.inverse_document_frequency = {items: (format(math.log(len(self.tweets) / self.document_frequency[items], 10), '.3f') \
                                                if self.document_frequency[items] != 0 else format(0, '.3f')) \
                                                    for items in self.document_frequency}
        self.inverse_document_frequency = AlphaSortedDict(self.inverse_document_frequency)
        if debug:
            print '\nInverse Document Frequency: ', self.inverse_document_frequency

    # calculate tf * idf of training data
    def calculate_tf_idf(self):
        self.tot_tf_idf = []
        for items in self.total_term_frequency:
            sep_tf_idf = {}
            sep_tf_idf = {item: float(self.inverse_document_frequency[item]) * items[0][item] for item in items[0]}
            sep_tf_idf = AlphaSortedDict(sep_tf_idf)
            self.tot_tf_idf.append((sep_tf_idf, items[1]))
        if debug:
            print '\nTF-IDF: ', self.tot_tf_idf, '\n\n'
        return self.tot_tf_idf

# tf = TrainingTfIdf()


from config import debug
import csv
import re
import MySQLdb as md
from nltk.stem.porter import *

# get training and testing data
class Data:
    # constructor (initially loads training and testing data)
    def __init__(self):
        if debug:
            print '\nData: (data.py)\n'

        self.stemmer = PorterStemmer()
        # training data
        cfile = open('new_training_filtered.csv')
    	cfile_read = csv.reader(cfile)
    	cfile_data = list(cfile_read)
    	self.cfile_data_lower = []
    	[self.cfile_data_lower.append([sentiment, words.lower()]) for [sentiment, words] in cfile_data]

        # testing data
        con = md.connect('localhost','root','12345','tweetdb')
    	cur = con.cursor()
    	cur.execute("select * from realtweetsentiment_realtweet")
    	tw = []
    	count = 0
    	for row in cur.fetchall():
    		tw.append(row[1])
    		count += 1
    	self.tw_lower = []
    	[self.tw_lower.append(items.lower()) for items in tw]

        self.helper_training()
        self.helper_testing()
        self.tot_bag_words()

    # filter training data
    def helper_training(self):
        self.tweets = self.cfile_data_lower[:10]
        if debug:
            print '\nLen of training: ', len(self.tweets)

        a = open('stopword.txt', 'r')
        stopwords = [i.replace('\n','') for i in a]

        # total words
        tot_words = (items for [sentiment, words] in self.tweets \
        		 	    for items in words.split())
        if debug:
            print '\nTotal training words: ', tot_words

        # remove punctuations and filter the words
        training_clean_words = ((re.sub(r'\ *\w*@\w*|\ *http\S*|&\w+|[^A-Za-z0-9\s+]+|\d*|\r\n?', '', items)) \
                                    .replace('\n',' ') for [sentiment, words] in self.tweets \
                                        for items in words.split())
        if debug:
            print '\nTraining clean words: ', training_clean_words

        # remove stop words from words
        after_training_stop_words = (items for items in training_clean_words if items not in stopwords)
        if debug:
            print '\nafter_training_stop_words: ', after_training_stop_words

        # perform stemming in words
        after_training_stemming = (self.stemmer.stem(items) for items in after_training_stop_words)
        if debug:
            print '\nafter_training_stemming: ', after_training_stemming

        # distinct words after stemming (bag_of_words)
        self.bag_of_words_train = list(set(items for items in after_training_stemming))
        self.bag_of_words_train.sort()
        if debug:
            print '\nself.bag_of_words_train: ', self.bag_of_words_train  

    # filter testing Data
    def helper_testing(self):
        test = self.tw_lower[:10]
        if debug:
            print '\nLen of tests: ', len(test)

        a = open('stopword.txt', 'r')
        stopwords = [i.replace('\n','') for i in a]

        # clean tweets in sentence level
        self.sentence_filtered = [(re.sub(r'\ *\w*@\w*|\ *http\S*|&\w+|[^A-Za-z0-9\s+]+|\d*|\r \n?', '', \
                                    sentence)).replace('\n',' ') for sentence in test]
        if debug:
            print '\nTweet sentence filtered: ', self.sentence_filtered

        # remove stop words
        after_testing_stop_words = (words for items in self.sentence_filtered for words in items.split() \
                                        if words not in stopwords)
        if debug:
            print '\nafter_testing_stop_words: ', after_testing_stop_words

        # perform stepping
        after_testing_stemming = (self.stemmer.stem(items) for items in after_testing_stop_words)
        if debug:
            print '\nafter_testing_stemming: ', after_testing_stemming

        self.bag_of_words_test = list(set(items for items in after_testing_stemming))
        self.bag_of_words_test.sort()
        if debug:
            print '\nbag_of_words_test: ', self.bag_of_words_test

    # total distinct words
    def tot_bag_words(self):
        # self.total_bag_words = []
        self.total_bag_words = list(set(items for items in self.bag_of_words_train + self.bag_of_words_test))
        self.total_bag_words.sort()
        if debug:
            print '\ntotal bag of words: ', self.total_bag_words, '\n\n'

# d = Data()

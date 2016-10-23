

import csv
import math
import re
import datetime
import MySQLdb as md

def get_words_in_training_data():
	cfile = open('new_training_filtered.csv')
	cfile_read = csv.reader(cfile)
	cfile_data = list(cfile_read)
	cfile_data_lower = []
	[cfile_data_lower.append([sentiment, words.lower()]) for [sentiment, words] in cfile_data]
	return cfile_data_lower[:1000]

def get_words_in_testing_data():
	con = md.connect('localhost','root','12345','realtweetdb')
	cur = con.cursor()
	cur.execute("select * from realtweetsentiment_realtweet")

	tw = []
	count = 0
	for row in cur.fetchall():
		tw.append(row[1])
		count += 1
	tw_lower = []
	[tw_lower.append(items.lower()) for items in tw]
	return tw_lower[:5]

def helper_training(tweets):
    a = open('stopword.txt', 'r')
    stopwords = [i.replace('\n','') for i in a]

    tot_words = [items for [sentiment, words] in tweets \
    		 	    for items in words.split()]

    bag_of_words_train = []
    [bag_of_words_train.append((re.sub(r'\ *\w*@\w*|\ *http\S*|&\w+|[^A-Za-z0-9\s+]+|\d*|\r\n?', '', \
		items)).replace('\n',' ')) \
			for [sentiment, words] in tweets \
				for items in words.split() \
					if items not in bag_of_words_train and items not in stopwords]

    bag_of_words_train.sort()
    print '\nTraining: '
    # print tweets
    print '\nTotal Words: ', len(tot_words)
    print '\nUnique Words: ', len(bag_of_words_train), '\n\nBag of Words: \n', bag_of_words_train, '\n'

    return [item for item in tweets], bag_of_words_train

def helper_testing(test):
    a = open('stopword.txt', 'r')
    stopwords = [i.replace('\n','') for i in a]

    tot_words = [items for words in test \
    		 	    for items in words.split()]

    sentence_filtered = []

    for sentence in test:
        b = []
        sentence = (re.sub(r'\ *\w*@\w*|\ *http\S*|&\w+|[^A-Za-z0-9\s+]+|\d*|\r \n?', '', sentence)).replace('\n',' ')
        a = sentence.split()
        for items in a:
            if items not in stopwords:
                b.append(items)
        sentence_filtered.append(' '.join(b))

    # print '\nTesting: '
    # print sentence_filtered

    return sentence_filtered

def calculate_tf(tweets, bag_of_words):
    total_term_frequency = []
    document_frequency = []
    for i, items in enumerate(tweets):
        term_frequency = []
        for item in bag_of_words:
            if item in items[1]:
                if item not in document_frequency:
                    count = 1
                    document_frequency.append((item, count))
                else:
                    count += 1
                    document_frequency.append((item, count))
                if item not in term_frequency:
                    count = 1
                    term_frequency.append((item, count))
                else:
                    count += 1
                    term_frequency.append((item, count))
            else:
                term_frequency.append((item, 0))
                # term_frequency[item] = 0
        # print 'Tweet ', i + 1, ': ' '\t', term_frequency, '\n'
        total_term_frequency.append((term_frequency, items[0]))
    # print 'Total term frquency: \n', total_term_frequency, '\n'
    # print 'Document Frequency: ', '\n', document_frequency, '\n'
    return tweets, document_frequency, total_term_frequency

def calculate_idf(tweets, document_frequency):
	inverse_document_frequency = \
		{items[0]: format(math.log(len(tweets) / items[1], 10), '.3f') \
			for items in document_frequency}
	# print 'Inverse Document Frequency: ', '\n', inverse_document_frequency, '\n'
	return inverse_document_frequency

def calculate_tf_idf(total_term_frequency, inverse_document_frequency):
    tf_idf, tot_tf_idf = (), []
    for items in total_term_frequency:
        sep_tf_idf = []
        for item in inverse_document_frequency:
            for tup in items[0]:
                if item in tup:
                    tf_idf = (item, float(inverse_document_frequency[item]) * tup[1])
                    sep_tf_idf.append(tf_idf)
        tot_tf_idf.append((sep_tf_idf, items[1]))
    print '\nTraining tf-idf: '
    for items in tot_tf_idf:
        print items, '\n'
    return tot_tf_idf

if __name__ == '__main__':
    start = datetime.datetime.now()

    a = get_words_in_training_data()
    b, c = helper_training(a)
    d, e, f = calculate_tf(b, c)
    g = calculate_idf(d, e)
    h = calculate_tf_idf(f, g)

    stop = datetime.datetime.now()

    # print '\nTraining tf-idf: \n\n', h, '\n'

    print '\nDone\n'

    print '\nTotal time taken: ', stop - start, '\n'

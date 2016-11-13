
from django.shortcuts import render
from django.views.generic import View
from computation.naive_bayes import NaiveBayes
from computation.accuracy import Accuracy
import subprocess
from .models import TestingResult, ValidationResult

# Create your views here.

class IndexView(View):
    template_name = 'twitterSentiment/base.html'

    def get(self, request):
        return render(request, self.template_name, {'message': 'Hello!! Twitter Sentiment Analysis!!!!'})

class ValidationView(View):
    template_name = 'twitterSentiment/training_phase_form.html'

    def get(self, request):
        len_tr = subprocess.check_output("cat new_training_filtered.csv | wc -l", shell=True)
        return render(request, self.template_name, {'len': len_tr})

    def post(self, request):
        trnum = request.POST.get('trnum')
        train_ratio = request.POST.get('ratio')

        try:
            row = ValidationResult.objects.get(no_of_train_data=trnum, train_split_ratio=train_ratio)
            print 'Row: ', row
            if row:
                print 'Inside row!'
                context = {
                    'tr_ratio' : row.train_split_ratio,
                    'te_ratio' : row.test_split_ratio,
                    'trnum' : row.no_of_train_data,
                    'trpos' : row.no_of_pos_train_data,
                    'trneg' : row.no_of_neg_train_data,
                    'tepos' : row.no_of_pos_test_data,
                    'teneg' : row.no_of_neg_test_data,
                    'cpos' : row.no_of_pos_classified,
                    'cneg' : row.no_of_neg_classified,
                    'apos' : row.no_of_actual_pos,
                    'aneg' : row.no_of_actual_neg,
                    'precision' : row.precision,
                    'recall' : row.recall,
                    'f_measure' : row.f_measure,
                    'accuracy' : row.accuracy
                }
        except:
            ac = Accuracy(trnum, train_ratio)
            pos_train, neg_train, pos_test, neg_test, classified_pos, classified_neg, actual_pos, actual_neg, precision, \
                recall, f_measure, accuracy = len(ac.posit_train_tfidf), len(ac.negat_train_tfidf), len(ac.pos_test), \
                    len(ac.neg_test), ac.count_pos, ac.count_neg, ac.actual_pos, ac.actual_neg, ac.avg_precision, ac.avg_recall, \
                        ac.f_measure, ac.accuracy
            ValidationResult.objects.create(
                train_split_ratio = train_ratio,
                test_split_ratio = 1 - float(train_ratio),
                no_of_train_data = trnum,
                no_of_pos_train_data = pos_train,
                no_of_neg_train_data = neg_train,
                no_of_pos_test_data = pos_test,
                no_of_neg_test_data = neg_test,
                no_of_pos_classified = classified_pos,
                no_of_neg_classified = classified_neg,
                no_of_actual_pos = actual_pos,
                no_of_actual_neg = actual_neg,
                precision = precision,
                recall = recall,
                f_measure = f_measure,
                accuracy = accuracy
            )
            context = {
                'tr_ratio' : train_ratio,
                'te_ratio' : 1 - float(train_ratio),
                'trpos' : pos_train,
                'trneg' : neg_train,
                'tepos' : pos_test,
                'teneg' : neg_test,
                'cpos' : classified_pos,
                'cneg' : classified_neg,
                'apos' : actual_pos,
                'aneg' : actual_neg,
                'precision' : precision,
                'recall' : recall,
                'f_measure' : f_measure,
                'accuracy' : accuracy
            }

        return render(request, 'twitterSentiment/training_phase_result.html', context)

class TestingView(View):
    template_name = 'twitterSentiment/testing_phase_form.html'

    def get(self, request):
        len_tr = subprocess.check_output("cat new_training_filtered.csv | wc -l", shell=True)
        len_te = subprocess.check_output("cat test_tweets.csv | wc -l", shell=True)
        return render(request, self.template_name, {'trlen': len_tr, 'telen': len_te})

    def post(self, request):
        trnum = request.POST.get('trnum')
        tenum = request.POST.get('tenum')

        try:
            row = TestingResult.objects.get(no_of_train_data=trnum, no_of_test_data=tenum)
            if row:
                context = {
                    'tr': row.no_of_train_data,
                    'te': row.no_of_test_data,
                    'pos': row.no_of_pos_classified,
                    'neg': row.no_of_neg_classified
                }
        except:
            nb = NaiveBayes(trnum, tenum)
            train_num, test_num, pos_num, neg_num = nb.trnum, nb.tenum, nb.pos, nb.neg
            TestingResult.objects.create(no_of_train_data = train_num, no_of_test_data = test_num, \
                                        no_of_pos_classified = pos_num, no_of_neg_classified = neg_num)
            context = {
                'tr' : train_num,
                'te' : test_num,
                'pos' : pos_num,
                'neg' : neg_num
            }

        return render(request, 'twitterSentiment/testing_phase_result.html', context)

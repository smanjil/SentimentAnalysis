from django.shortcuts import render
from django.views.generic import View
from computation.naive_bayes import NaiveBayes
import subprocess
from .models import TestingResult

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
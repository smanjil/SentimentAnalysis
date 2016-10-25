from django.shortcuts import render
from django.views.generic import View
from computation.data import Data
import subprocess

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

        data = Data(trnum, tenum)

        return render(request, 'twitterSentiment/testing_phase_result.html', {'tr': trnum, 'te':tenum})
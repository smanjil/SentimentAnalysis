from django.shortcuts import render
from django.views.generic import View


# Create your views here.

class IndexView(View):
    template_name = 'twitterSentiment/base.html'

    def get(self, request):
        return render(request, self.template_name, {'message': 'Hello!! Twitter Sentiment Analysis!!!!'})

class ValidationView(View):
    template_name = 'twitterSentiment/training_phase_form.html'

    def get(self, request):
        return render(request, self.template_name, {'message': 'Fill in the required fields!!'})

class TestingView(View):
    template_name = 'twitterSentiment/testing_phase_form.html'

    def get(self, request):
        return render(request, self.template_name, {'message': 'Fill in the required fields!!'})
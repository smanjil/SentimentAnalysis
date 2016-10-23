from django.shortcuts import render
from django.views.generic import View


# Create your views here.

class IndexView(View):
    template_name = 'twitterSentiment/index.html'

    def get(self, request):
        return render(request, self.template_name, {'message': 'Hello!! Twitter Sentiment Analysis!!!!'})

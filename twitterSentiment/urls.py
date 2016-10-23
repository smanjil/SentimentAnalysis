
from django.conf.urls import url
from . import views

app_name = 'twitterSentiment'

urlpatterns = [
    # homepage
    url(r'^$', views.IndexView.as_view(), name='index'),
]
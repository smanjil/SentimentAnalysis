
from django.conf.urls import url
from . import views

app_name = 'twitterSentiment'

urlpatterns = [
    # homepage
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^training/$', views.ValidationView.as_view() , name='training'),
    url(r'^testing/$', views.TestingView.as_view() , name='testing'),
]
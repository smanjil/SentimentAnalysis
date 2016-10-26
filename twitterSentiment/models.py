from __future__ import unicode_literals

from django.db import models

# Create your models here.

class StoreTrainingFile(models.Model):
    fname = models.CharField(max_length=50)

    def __str__(self):
        return self.fname

class StoreTestingFile(models.Model):
    fname = models.CharField(max_length=50)

    def __str__(self):
        return self.fname

class TweetCollection(models.Model):
    tweet_text = models.TextField()
    tweet_date = models.DateTimeField()

    def __str__(self):
        return self.tweet_text

class TestingResult(models.Model):
    no_of_train_data = models.IntegerField()
    no_of_test_data = models.IntegerField()
    no_of_pos_classified = models.IntegerField()
    no_of_neg_classified = models.IntegerField()

    def __str__(self):
        string = str(self.no_of_train_data) + ' ' + str(self.no_of_test_data)
        return string

class ValidationResult(models.Model):
    train_split_ratio = models.FloatField()
    test_split_ratio = models.FloatField()
    no_of_pos_train_data = models.IntegerField()
    no_of_neg_train_data = models.IntegerField()
    no_of_pos_test_data = models.IntegerField()
    no_of_neg_test_data = models.IntegerField()
    no_of_pos_classified = models.IntegerField()
    no_of_neg_classified = models.IntegerField()
    no_of_actual_pos = models.IntegerField()
    no_of_actual_neg = models.IntegerField()
    precision = models.FloatField()
    recall = models.FloatField()
    f_measure = models.FloatField()
    accuracy = models.FloatField()
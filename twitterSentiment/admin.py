from django.contrib import admin
from .models import StoreTrainingFile, StoreTestingFile, TestingResult, ValidationResult, TweetCollection

# Register your models here.

admin.site.register(StoreTrainingFile)
admin.site.register(StoreTestingFile)
admin.site.register(TestingResult)
admin.site.register(ValidationResult)
admin.site.register(TweetCollection)
from django.urls import path
from . import views

app_name = 'text_model'

urlpatterns = [
    path('',views.text,name='text'),
    path('sentiment/',views.sentiment,name='sentiment'),
    path('sent_results/',views.training,name='sentiment_result'),
    path('download/',views.download_models,name='download'),
]
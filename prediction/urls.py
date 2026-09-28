from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('apply/', views.apply, name='apply'),
    path('result/<int:pk>/', views.prediction_result, name='prediction_result'),
    path('history/', views.prediction_history, name='prediction_history'),
]

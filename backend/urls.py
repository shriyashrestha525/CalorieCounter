from django.urls import path
from .views import predict_food  # Import the view

urlpatterns = [
    path('predict-food/', predict_food, name='predict_food'),  # Define the URL
]

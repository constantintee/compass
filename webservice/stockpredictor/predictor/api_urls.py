# predictor/api_urls.py (Optional - if you want to add API endpoints)
from django.urls import path
from . import api_views

app_name = 'predictor-api'

urlpatterns = [
    path('predictions/<str:ticker>/', api_views.PredictionView.as_view(), name='prediction'),
    path('stocks/top/', api_views.TopStocksView.as_view(), name='top-stocks'),
    path('technical/<str:ticker>/', api_views.TechnicalIndicatorsView.as_view(), name='technical'),
]
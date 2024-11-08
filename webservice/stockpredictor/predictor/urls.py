# predictor/urls.py
from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),
    
    # Prediction endpoints
    path('predict/<str:ticker>/', views.get_prediction, name='get_prediction'),
    path('predict/<str:ticker>/refresh/', views.refresh_prediction, name='refresh_prediction'),
    path('predict/<str:ticker>/graph/', views.get_prediction_graph, name='prediction_graph'),
    path('predict/<str:ticker>/history/', views.prediction_history, name='prediction_history'),
    
    # Top stocks
    path('stocks/top/', views.get_top_stocks, name='top_stocks'),
    
    # Technical analysis
    path('stocks/<str:ticker>/indicators/', views.get_technical_indicators, name='technical_indicators'),
    path('stocks/<str:ticker>/stats/', views.market_stats, name='market_stats'),
    
    # Search
    path('search/', views.stock_search, name='stock_search'),
]
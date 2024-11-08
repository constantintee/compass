# webservice/stockpredictor/predictor/apps.py
from django.apps import AppConfig

class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stockpredictor.predictor'
    verbose_name = 'Stock Predictor'

    def ready(self):
        try:
            import stockpredictor.predictor.signals
        except ImportError:
            pass
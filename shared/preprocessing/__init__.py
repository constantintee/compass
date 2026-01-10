# shared/preprocessing/__init__.py
"""
Preprocessing module for Compass Stock Prediction System.

This module provides data validation, cleaning, and transformation
functionality shared across all components.
"""

from .validators import DataValidator
from .cleaners import DataCleaner

# Lazy imports for modules with heavy dependencies (tensorflow, h5py)
def __getattr__(name):
    if name == 'DataTransformer':
        from .transformers import DataTransformer
        return DataTransformer
    elif name == 'Preprocessor':
        from .preprocessor import Preprocessor
        return Preprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'Preprocessor',
    'DataValidator',
    'DataCleaner',
    'DataTransformer',
]

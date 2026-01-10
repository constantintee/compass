# shared/config/__init__.py
"""
Configuration module for Compass Stock Prediction System.
"""

from .loader import load_configuration, ConfigLoader

__all__ = ['load_configuration', 'ConfigLoader']

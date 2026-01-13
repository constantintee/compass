# shared/config/loader.py
"""
Configuration loading utilities.
"""

import os
import sys
from typing import Any, Dict, Optional

import yaml

from ..exceptions import ConfigurationError


class ConfigLoader:
    """Configuration loader with environment variable support."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable overrides."""
        if self._config is not None:
            return self._config

        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Override sensitive data with environment variables if present
            if 'databasepsql' in config:
                config['databasepsql']['host'] = os.getenv(
                    'DB_HOST', config['databasepsql'].get('host')
                )
                config['databasepsql']['port'] = os.getenv(
                    'DB_PORT', config['databasepsql'].get('port')
                )
                config['databasepsql']['user'] = os.getenv(
                    'DB_USER', config['databasepsql'].get('user')
                )
                config['databasepsql']['password'] = os.getenv(
                    'DB_PASSWORD', config['databasepsql'].get('password')
                )
                config['databasepsql']['dbname'] = os.getenv(
                    'DB_NAME', config['databasepsql'].get('dbname')
                )

            self._config = config
            return config

        except FileNotFoundError:
            raise ConfigurationError(
                config_key='config_path',
                message=f"Configuration file not found: {self.config_path}"
            )
        except yaml.YAMLError as e:
            raise ConfigurationError(
                config_key='yaml',
                message=f"Error parsing YAML configuration: {e}"
            )
        except Exception as e:
            raise ConfigurationError(
                config_key='unknown',
                message=f"Failed to load configuration: {e}"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        config = self.load()
        return config.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get a nested configuration value."""
        config = self.load()
        value = config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value


def load_configuration(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    This is a convenience function that maintains backwards compatibility.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.load()

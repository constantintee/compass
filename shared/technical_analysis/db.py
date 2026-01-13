# shared/technical_analysis/db.py
"""
Database connection management for technical analysis.
"""

import os
import logging
from typing import Optional

import psycopg2
from psycopg2 import pool, OperationalError

from ..constants import DatabaseConfig
from ..exceptions import DatabaseError


class DBConnectionManager:
    """Singleton database connection pool manager."""

    _instance: Optional['DBConnectionManager'] = None
    _pool: Optional[pool.SimpleConnectionPool] = None

    def __new__(cls) -> 'DBConnectionManager':
        if cls._instance is None:
            cls._instance = super(DBConnectionManager, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance

    def _initialize_pool(self) -> None:
        """Initialize the database connection pool."""
        if self._pool is None:
            try:
                self._pool = pool.SimpleConnectionPool(
                    minconn=DatabaseConfig.MIN_CONNECTIONS,
                    maxconn=DatabaseConfig.MAX_CONNECTIONS,
                    host=os.getenv('DB_HOST'),
                    port=os.getenv('DB_PORT'),
                    user=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASSWORD'),
                    dbname=os.getenv('DB_NAME')
                )
            except OperationalError as e:
                raise DatabaseError(
                    operation='initialize_pool',
                    message=f"Failed to initialize database connection pool: {e}"
                )

    def get_connection(self) -> psycopg2.extensions.connection:
        """Get a connection from the pool."""
        if self._pool is None:
            raise DatabaseError(
                operation='get_connection',
                message="Connection pool not initialized"
            )
        return self._pool.getconn()

    def return_connection(self, conn: psycopg2.extensions.connection) -> None:
        """Return a connection to the pool."""
        if self._pool is not None:
            self._pool.putconn(conn)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        if cls._instance is not None:
            cls._instance.close_all()
            cls._instance = None

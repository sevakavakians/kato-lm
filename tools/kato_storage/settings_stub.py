"""
Minimal settings stub for portable notebook usage.

This provides just enough configuration for the connection_manager to work
without requiring the full KATO settings infrastructure.
"""

import os
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration."""
    # Use environment variables or default to localhost for host machine access
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', '6379'))
    redis_enabled: bool = True

    # ClickHouse settings
    CLICKHOUSE_HOST: str = os.getenv('CLICKHOUSE_HOST', 'localhost')
    CLICKHOUSE_PORT: int = int(os.getenv('CLICKHOUSE_PORT', '8123'))
    CLICKHOUSE_DB: str = os.getenv('CLICKHOUSE_DB', 'kato')

    @property
    def REDIS_URL(self):
        """Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    @property
    def redis_url(self):
        """Backward compatibility property."""
        return self.REDIS_URL

    @property
    def clickhouse_host(self):
        return self.CLICKHOUSE_HOST

    @property
    def clickhouse_port(self):
        return self.CLICKHOUSE_PORT

    @property
    def clickhouse_db(self):
        return self.CLICKHOUSE_DB


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = 'INFO'


@dataclass
class Settings:
    """Minimal settings for connection manager."""
    database: DatabaseConfig
    logging: LoggingConfig

    def __init__(self):
        self.database = DatabaseConfig()
        self.logging = LoggingConfig()


# Singleton instance
_settings = None


def get_settings():
    """Get settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

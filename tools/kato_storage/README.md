# KATO Storage Utilities - Portable Copy

This directory contains portable copies of KATO storage utilities for use in Jupyter notebooks.

## Required Python Dependencies

For these utilities to work, your Jupyter environment needs the following packages:

```bash
pip install clickhouse-connect redis datasketch
```

Or in a Jupyter notebook cell:
```python
!pip install clickhouse-connect redis datasketch
```

## Files

- `__init__.py` - Package initialization
- `connection_manager.py` - Database connection utilities
- `clickhouse_writer.py` - ClickHouse pattern data storage
- `redis_writer.py` - Redis metadata storage
- `settings_stub.py` - Minimal configuration (replaces full KATO settings)

## Connection Settings

Default container names (can be overridden via environment variables):
- **MongoDB**: `mongodb://kato-mongodb:27017`
- **Redis**: `redis://kato-redis:6379/0`
- **ClickHouse**: `kato-clickhouse:8123` (database: `kato`)

## Usage

```python
from tools.kato_storage import (
    get_clickhouse_client,
    get_redis_client,
    ClickHouseWriter,
    RedisWriter
)

# Get clients
clickhouse_client = get_clickhouse_client()
redis_client = get_redis_client()

# Initialize writers
clickhouse_writer = ClickHouseWriter(kb_id='node0', clickhouse_client=clickhouse_client)
redis_writer = RedisWriter(kb_id='node0', redis_client=redis_client)
```

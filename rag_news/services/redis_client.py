"""Singleton Redis client helper."""

from __future__ import annotations

import threading
from typing import Optional

import redis

from .. import config

_client: Optional[redis.Redis] = None
_lock = threading.Lock()


def get_client() -> redis.Redis:
    """Return a thread-safe, lazily initialised Redis client."""

    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
    return _client

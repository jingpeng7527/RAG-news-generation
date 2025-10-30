"""Redis-backed caching primitives for Congress.gov payloads."""

from __future__ import annotations

import json
from typing import Any, Optional

from .redis_client import get_client
from .. import config


class BillCache:
    """Cache layer for bill payloads."""

    def __init__(self) -> None:
        self._redis = get_client()
        self._ttl = config.CACHE_TTL_SECONDS

    def _key(self, bill_id: str) -> str:
        return f"bill-cache:{bill_id.lower()}"

    def fetch(self, bill_id: str) -> Optional[dict[str, Any]]:
        raw = self._redis.get(self._key(bill_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def store(self, bill_id: str, payload: dict[str, Any]) -> None:
        self._redis.setex(self._key(bill_id), self._ttl, json.dumps(payload))

"""Congress.gov API client with caching and vector store integration."""

from __future__ import annotations

import logging
from typing import Any, Dict

import requests

from .. import config
from .cache import BillCache
from .vector_store import BillVectorStore, build_context_chunks


log = logging.getLogger(__name__)


class CongressAPI:
    """Fetches and caches bill data while refreshing the vector store."""

    def __init__(self, cache: BillCache | None = None, store: BillVectorStore | None = None, session: requests.Session | None = None) -> None:
        self._cache = cache or BillCache()
        self._store = store or BillVectorStore()
        self._session = session or requests.Session()

    def fetch_bill_bundle(self, bill_id: str, bill_type: str, congress: int) -> Dict[str, Any]:
        cached = self._cache.fetch(bill_id)
        if cached:
            return cached

        bundle = self._download_bundle(bill_id, bill_type, congress)
        if bundle and bundle.get("bill"):
            self._cache.store(bill_id, bundle)

            contexts = build_context_chunks(bundle)
            if contexts:
                self._store.upsert_bill(bill_id, contexts)
        else:
            log.warning("Congress API returned empty payload for %s", bill_id)

        return bundle

    def _download_bundle(self, bill_id: str, bill_type: str, congress: int) -> Dict[str, Any]:
        number = _extract_bill_number(bill_id)
        base = f"{config.CONGRESS_API_BASE_URL}/bill/{congress}/{bill_type}/{number}"

        bill = self._request(f"{base}?api_key={config.CONGRESS_API_KEY}")
        if not bill:
            return {}

        for resource in ("summaries", "actions", "committees", "amendments", "cosponsors"):
            payload = self._request(f"{base}/{resource}?api_key={config.CONGRESS_API_KEY}")
            if payload:
                bill["bill"][resource] = payload.get(resource, payload)
            else:
                bill["bill"][resource] = {}

        return bill

    def _request(self, url: str) -> Dict[str, Any]:
        try:
            response = self._session.get(url, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:  # pragma: no cover - network failures
            log.warning("Congress API request failed: %s", exc)
            return {}

    def query_context(self, bill_id: str, question: str) -> list[str]:
        return self._store.query(bill_id, question)

    def refresh_vector_store(self, bill_id: str, payload: dict) -> None:
        contexts = build_context_chunks(payload)
        self._store.upsert_bill(bill_id, contexts)


def _extract_bill_number(bill_id: str) -> str:
    return "".join(filter(str.isdigit, bill_id))

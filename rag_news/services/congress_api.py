"""Congress.gov API client with caching and vector store integration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

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

        for resource in config.CONGRESS_BILL_RESOURCES:
            resource = resource.strip().strip("/")
            if not resource:
                continue

            payload = self._request(f"{base}/{resource}?api_key={config.CONGRESS_API_KEY}")
            key = resource.replace("/", "_")
            if payload:
                bill["bill"][key] = self._extract_resource_payload(payload, resource, key)
            else:
                bill["bill"][key] = {}

        self._fetch_additional_resources(bill, bill_id, bill_type, congress, number)

        return bill

    def _fetch_additional_resources(
        self,
        bundle: Dict[str, Any],
        bill_id: str,
        bill_type: str,
        congress: int,
        number: str,
    ) -> None:
        resources = config.CONGRESS_ADDITIONAL_ENDPOINTS
        if not resources:
            return

        bill = bundle.get("bill")
        if not isinstance(bill, dict):
            return

        base_context = {
            "bill_id": bill_id,
            "bill_type": bill_type,
            "bill_number": number,
            "congress": congress,
        }

        for definition in resources:
            if not isinstance(definition, dict):
                continue
            alias = definition.get("alias")
            path_template = definition.get("path")
            if not alias or not path_template:
                continue

            source_path = definition.get("source", "")
            entries = self._resolve_source_entries(bundle, source_path)
            if not entries:
                entries = [bill]

            max_items = definition.get("max")
            if isinstance(max_items, int) and max_items > 0:
                entries = entries[:max_items]

            field_map = definition.get("field_map")
            collected: List[Any] = []
            for entry in entries:
                context = self._build_format_context(base_context, entry, field_map)
                try:
                    formatted_path = path_template.format(**context)
                except KeyError as exc:
                    log.debug("Skipping %s endpoint due to missing field: %s", alias, exc)
                    continue

                url = f"{config.CONGRESS_API_BASE_URL}/{formatted_path.strip('/')}"
                payload = self._request(f"{url}?api_key={config.CONGRESS_API_KEY}")
                if payload:
                    collected.append(payload)

            if collected:
                bill[alias] = collected

    def _extract_resource_payload(self, payload: Dict[str, Any], resource: str, key: str) -> Dict[str, Any] | Any:
        if isinstance(payload, dict):
            for candidate in (resource, key):
                if candidate in payload:
                    return payload[candidate]
        return payload

    def _resolve_source_entries(self, bundle: Dict[str, Any], source_path: str) -> List[Dict[str, Any]]:
        if not source_path:
            bill = bundle.get("bill")
            return [bill] if isinstance(bill, dict) else []

        current: Any = bundle
        for part in source_path.split("."):
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return []

        if isinstance(current, list):
            return [item for item in current if isinstance(item, dict)]
        if isinstance(current, dict):
            nested_items: List[Dict[str, Any]] = []
            for candidate in ("items", "item", "data", "results"):
                value = current.get(candidate)
                if isinstance(value, list):
                    nested_items.extend([elem for elem in value if isinstance(elem, dict)])
            if nested_items:
                return nested_items
            return [current]
        return []

    def _build_format_context(
        self,
        base_context: Dict[str, Any],
        entry: Dict[str, Any] | Any,
        field_map: Dict[str, str] | None,
    ) -> Dict[str, Any]:
        context = dict(base_context)
        if isinstance(entry, dict):
            for key, value in entry.items():
                if isinstance(key, str) and not isinstance(value, (list, dict)):
                    context[key] = value

        if isinstance(field_map, dict):
            for placeholder, source_key in field_map.items():
                value = None
                if isinstance(entry, dict):
                    value = self._lookup_nested(entry, source_key)
                if value is None:
                    value = self._lookup_nested(context, source_key)
                if value is not None:
                    context[placeholder] = value

        return context

    def _lookup_nested(self, data: Any, path: str) -> Any:
        current = data
        for part in path.split("."):
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

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

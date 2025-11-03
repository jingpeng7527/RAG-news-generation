"""Worker that checks hyperlinks before publishing."""

from __future__ import annotations

import json
import re
import threading
from typing import Dict, List

import requests
from kafka import KafkaConsumer

from .. import config
from ..services import RedisStateStore


class LinkCheckWorker(threading.Thread):
    """Validates hyperlinks inside generated Markdown content."""

    URL_PATTERN = re.compile(r"http[s]?://[^ )]+")

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.consumer = KafkaConsumer(
            config.KAFKA_TOPICS["link_check"],
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="rag-link-workers",
            enable_auto_commit=True,
            auto_offset_reset="earliest",
        )
        self.state = RedisStateStore()

    def run(self) -> None:  # pragma: no cover - infinite loop
        print("[link] worker ready")
        for message in self.consumer:
            payload: Dict = message.value
            article_text = payload["article_text"]
            bill_payload = payload["bill_payload"]
            bill_id = payload["bill_id"]

            # Validate links (results logged but not blocking)
            invalid_links = self._validate_links(article_text)
            if invalid_links:
                print(f"[link] warning: {len(invalid_links)} invalid link(s) found for {bill_id.upper()}")

            bill = bill_payload.get("bill", {}) if isinstance(bill_payload, dict) else {}
            output_record = {
                "bill_id": bill_id,
                "bill_title": bill.get("title", "Unknown"),
                "sponsor_bioguide_id": _extract_sponsor_id(bill),
                "bill_committee_ids": _committee_codes(bill),
                "article_content": article_text,
            }

            self.state.append_article_bundle(output_record)
            print(f"[link] stored article for {bill_id.upper()}")

    def _validate_links(self, text: str) -> List[str]:
        invalid: List[str] = []
        for url in self.URL_PATTERN.findall(text):
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                if response.status_code >= 400:
                    invalid.append(url)
            except requests.RequestException:
                invalid.append(url)
        return invalid


def _extract_sponsor_id(bill: dict) -> str:
    sponsors = bill.get("sponsors", [])
    if sponsors:
        return sponsors[0].get("bioguideId", "N/A")
    return "N/A"


def _committee_codes(bill: dict) -> list[str]:
    committees_section = bill.get("committees")
    committees = _extract_committee_items(committees_section)
    codes: list[str] = []
    for committee in committees:
        code = committee.get("systemCode") or committee.get("code")
        if code:
            codes.append(code)
    return codes


def _extract_committee_items(section) -> list[dict]:
    if isinstance(section, list):
        return [item for item in section if isinstance(item, dict)]

    if isinstance(section, dict):
        for key in ("committees", "items"):
            if key in section:
                items = _extract_committee_items(section[key])
                if items:
                    return items
    return []

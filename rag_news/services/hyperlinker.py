"""Utilities for enriching generated articles with relevant hyperlinks."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .. import config


def add_hyperlinks(article_text: str, bill_payload: dict) -> str:
    """Inject Markdown hyperlinks into the generated article when possible."""

    bill = bill_payload.get("bill", {}) if isinstance(bill_payload, dict) else {}
    text = article_text

    text = _link_bill_reference(text, bill)
    text = _link_people(text, bill)
    text = _link_committees(text, bill)
    return text


def _link_bill_reference(text: str, bill: dict) -> str:
    bill_type = bill.get("type") or bill.get("billType")
    number = bill.get("number") or bill.get("billNumber")
    congress = bill.get("congress")
    if not (bill_type and number and congress):
        return text

    formatted = f"{bill_type.upper()}{number}"
    url = f"https://www.congress.gov/bill/{congress}th-congress/{_bill_path_slug(bill_type)}/{number}"
    url = _append_api_key(url)

    pattern = re.compile(rf"\b{re.escape(formatted)}\b", re.IGNORECASE)
    return pattern.sub(f"[{formatted}]({url})", text)


def _link_people(text: str, bill: dict) -> str:
    for person in bill.get("sponsors", []):
        full_name = person.get("fullName")
        url = _normalize_url(person.get("url"))
        if full_name and url:
            text = text.replace(full_name, f"[{full_name}]({url})")

    cosponsors = bill.get("cosponsors", {})
    cosponsor_items = cosponsors.get("items") if isinstance(cosponsors, dict) else cosponsors
    for person in cosponsor_items or []:
        full_name = person.get("fullName")
        url = _normalize_url(person.get("url"))
        if full_name and url:
            text = text.replace(full_name, f"[{full_name}]({url})")
    return text


def _link_committees(text: str, bill: dict) -> str:
    committees_section = bill.get("committees")
    committees = []
    if isinstance(committees_section, dict):
        committees = committees_section.get("committees", {}).get("items", [])
        if not committees:
            committees = committees_section.get("items", [])
    elif isinstance(committees_section, list):
        committees = committees_section

    for committee in committees or []:
        name = committee.get("name")
        url = _normalize_url(committee.get("url"))
        if name and url:
            text = text.replace(name, f"[{name}]({url})")
    return text


def _normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    return _append_api_key(url)


def _append_api_key(url: str) -> str:
    api_key = config.CONGRESS_API_KEY
    parsed = urlparse(url)

    if "api.congress.gov" not in (parsed.netloc or "") or not api_key:
        # Strip existing query parameters for non-API links for consistency.
        return urlunparse(parsed._replace(query="")) if parsed.query else url

    query_pairs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_pairs["api_key"] = api_key
    new_query = urlencode(query_pairs)
    return urlunparse(parsed._replace(query=new_query))


def _bill_path_slug(bill_type: str | None) -> str:
    if not bill_type:
        return "bill"

    slug_map = {
        "hr": "house-bill",
        "s": "senate-bill",
        "hres": "house-resolution",
        "sres": "senate-resolution",
    }
    return slug_map.get(bill_type.lower(), f"{bill_type.lower()}-bill")

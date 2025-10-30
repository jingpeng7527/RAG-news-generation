"""Centralised configuration for the reimagined RAG News pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# --- Kafka Configuration ---------------------------------------------------

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

KAFKA_TOPICS: dict[str, str] = {
    "query": os.getenv("KAFKA_QUERY_TOPIC", "query.input"),
    "article": os.getenv("KAFKA_ARTICLE_TOPIC", "article.input"),
    "link_check": os.getenv("KAFKA_LINK_CHECK_TOPIC", "link.check.input"),
    "error": os.getenv("KAFKA_ERROR_TOPIC", "error.output"),
}

# --- Congress.gov API ------------------------------------------------------

CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY", "")
CONGRESS_API_BASE_URL = "https://api.congress.gov/v3"

# --- Redis -----------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 hours

# --- Vector Store ----------------------------------------------------------

VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "./vector_store")).resolve()
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))

# --- Output artefacts ------------------------------------------------------

OUTPUT_DIRECTORY = Path(os.getenv("OUTPUT_DIRECTORY", "output")).resolve()
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

_articles_filename = os.getenv("ARTICLES_FILENAME") or "articles.json"
OUTPUT_ARTICLES_FILE = OUTPUT_DIRECTORY / _articles_filename
OUTPUT_ANSWERS_FILE = OUTPUT_DIRECTORY / os.getenv("ANSWERS_FILENAME", "answers.json")


# --- Bills and Questions ---------------------------------------------------

TARGET_BILLS = [
    {"id": "hr1", "type": "hr", "congress": 119},
    {"id": "hr5371", "type": "hr", "congress": 119},
    {"id": "hr5401", "type": "hr", "congress": 119},
    {"id": "s2296", "type": "s", "congress": 119},
    {"id": "s24", "type": "s", "congress": 119},
    {"id": "s2882", "type": "s", "congress": 119},
    {"id": "s499", "type": "s", "congress": 119},
    {"id": "sres412", "type": "sres", "congress": 119},
    {"id": "hres353", "type": "hres", "congress": 119},
    {"id": "hr1968", "type": "hr", "congress": 119},
]

QUESTIONS = {
    1: "What does this bill do? Where is it in the process?",
    2: "What committees is this bill in?",
    3: "Who is the sponsor?",
    4: "Who cosponsored this bill? Are any of the cosponsors on the committee that the bill is in?",
    5: "Have any hearings happened on the bill? If so, what were the findings?",
    6: "Have any amendments been proposed on the bill? If so, who proposed them and what do they do?",
    7: "Have any votes happened on the bill? If so, was it a party-line vote or a bipartisan one?",
}


# --- Worker configuration --------------------------------------------------

NUM_QUERY_WORKERS = int(os.getenv("NUM_QUERY_WORKERS", "4"))
NUM_ARTICLE_WORKERS = int(os.getenv("NUM_ARTICLE_WORKERS", "2"))


# --- LLM Configuration -----------------------------------------------------

LLM_HOST = os.getenv("LLM_HOST", "127.0.0.1")
LLM_PORT = os.getenv("LLM_PORT", "11434")
LLM_API_BASE = f"http://{LLM_HOST}:{LLM_PORT}/api"
LLM_MODEL = os.getenv("LLM_MODEL", "gemma:2b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def build_topic_config() -> dict[str, Any]:
    """Return a copy of the Kafka topic mapping."""

    return dict(KAFKA_TOPICS)

"""RAG News package bundling configuration, services, and worker processes."""

from .config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPICS,
    QUESTIONS,
    TARGET_BILLS,
    OUTPUT_ARTICLES_FILE,
    OUTPUT_ANSWERS_FILE,
)

__all__ = [
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_TOPICS",
    "QUESTIONS",
    "TARGET_BILLS",
    "OUTPUT_ARTICLES_FILE",
    "OUTPUT_ANSWERS_FILE",
]

"""Tests for the reconstructed RAG News system."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Ensure the application package is importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide lightweight Kafka stubs when kafka-python is unavailable.
if "redis" not in sys.modules:  # pragma: no cover - optional dependency guard
    class _RedisStub:
        @classmethod
        def from_url(cls, *args, **kwargs):
            return MagicMock()

    sys.modules["redis"] = SimpleNamespace(Redis=_RedisStub)

if "kafka" not in sys.modules:  # pragma: no cover - import-time guard
    sys.modules["kafka"] = SimpleNamespace(
        KafkaConsumer=MagicMock,
        KafkaProducer=MagicMock,
    )
    sys.modules["kafka.admin"] = SimpleNamespace(KafkaAdminClient=MagicMock, NewTopic=MagicMock)
    sys.modules["kafka.errors"] = SimpleNamespace(NoBrokersAvailable=Exception)

from rag_news import config
from rag_news.services.congress_api import CongressAPI
from rag_news.services.state_store import RedisStateStore
from rag_news.services.vector_store import build_context_chunks
from rag_news.services.hyperlinker import add_hyperlinks
from rag_news.workers.question_worker import QuestionWorker
from rag_news.workers.link_worker import LinkCheckWorker


@dataclass
class FakeRedis:
    hashes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    sets: Dict[str, set] = field(default_factory=dict)
    strings: Dict[str, str] = field(default_factory=dict)

    def hset(self, key: str, *args, **kwargs) -> None:
        if args and isinstance(args[0], dict):
            mapping = args[0]
        elif "mapping" in kwargs:
            mapping = kwargs["mapping"]
        elif len(args) >= 2:
            mapping = {str(args[0]): args[1]}
        else:
            mapping = {}
        self.hashes.setdefault(key, {}).update({str(k): str(v) for k, v in mapping.items()})

    def hgetall(self, key: str) -> Dict[str, str]:
        return dict(self.hashes.get(key, {}))

    def sadd(self, key: str, value: Any) -> None:
        self.sets.setdefault(key, set()).add(int(value))

    def scard(self, key: str) -> int:
        return len(self.sets.get(key, set()))

    def delete(self, key: str) -> None:
        self.hashes.pop(key, None)
        self.sets.pop(key, None)
        self.strings.pop(key, None)

    def setex(self, key: str, ttl: int, value: str) -> None:  # pragma: no cover - simple store
        self.strings[key] = value

    def get(self, key: str) -> str | None:
        return self.strings.get(key)


@pytest.fixture(autouse=True)
def patch_redis(monkeypatch, tmp_path):
    fake = FakeRedis()
    monkeypatch.setattr("rag_news.services.redis_client.get_client", lambda: fake)
    monkeypatch.setattr("rag_news.services.state_store.get_client", lambda: fake)
    articles_path = tmp_path / "articles.json"
    monkeypatch.setattr(config, "OUTPUT_ARTICLES_FILE", articles_path)
    monkeypatch.setattr(config, "OUTPUT_ANSWERS_FILE", tmp_path / "answers.json")
    yield fake


def test_smoke_imports():
    """Basic smoke check that key modules load without side effects."""

    import rag_news.main  # noqa: F401
    import rag_news.services.congress_api  # noqa: F401
    import rag_news.workers.article_worker  # noqa: F401
    import rag_news.workers.link_worker  # noqa: F401


def test_state_store_round_trip(patch_redis):
    store = RedisStateStore()
    store.record_answer("hr1", 1, "Who sponsors?", "It is sponsored by Jane Doe.")
    assert patch_redis.hashes["bill:hr1:answers"] == {"1": "It is sponsored by Jane Doe."}
    assert store.fetch_answers("hr1") == {1: "It is sponsored by Jane Doe."}
    assert store.all_questions_answered("hr1") is False
    store.record_answer("hr1", 2, "What committees?", "Budget Committee")
    assert store.fetch_answers("hr1")[2] == "Budget Committee"


def test_congress_api_prefers_cache():
    dummy_payload = {"bill": {"title": "Sample"}}
    api = CongressAPI(cache=MagicMock(), store=MagicMock())
    api._cache.fetch.return_value = dummy_payload
    result = api.fetch_bill_bundle("hr1", "hr", 119)
    assert result == dummy_payload
    api._cache.store.assert_not_called()


def test_build_context_chunks_creates_meaningful_sections():
    sample = {
        "bill": {
            "title": "Test Bill",
            "summaries": {"items": [{"text": "A short summary."}]},
            "actions": {"items": [{"text": "Introduced in House."}]},
            "committees": [{"name": "Budget"}],
            "sponsors": [{"fullName": "Jane Doe", "party": "R"}],
        }
    }
    chunks = build_context_chunks(sample)
    assert any("Title" in chunk for chunk in chunks)
    assert any("Summary" in chunk for chunk in chunks)


def test_build_context_chunks_handles_list_sections():
    sample = {
        "bill": {
            "summaries": [
                {"text": "Direct list entry."}
            ],
            "committees": {
                "committees": {
                    "items": [
                        {"name": "Budget"}
                    ]
                }
            },
            "votes": [],
        }
    }
    chunks = build_context_chunks(sample)
    assert any("Direct list entry" in chunk for chunk in chunks)
    assert any("Committee: Budget" in chunk for chunk in chunks)


def test_add_hyperlinks_enriches_article(monkeypatch):
    monkeypatch.setattr(config, "CONGRESS_API_KEY", "TEST_KEY")
    article = "HR123 was introduced by Jane Doe in the Budget Committee."
    payload = {
        "bill": {
            "type": "hr",
            "number": "123",
            "congress": 119,
            "sponsors": [{"fullName": "Jane Doe", "url": "https://example.com/jane"}],
            "committees": {
                "committees": {
                    "items": [{"name": "Budget Committee", "url": "https://example.com/budget"}]
                }
            },
        }
    }
    enriched = add_hyperlinks(article, payload)
    assert "[HR123](https://www.congress.gov/bill/119th-congress/house-bill/123)" in enriched
    assert "[Jane Doe](https://example.com/jane)" in enriched
    assert "[Budget Committee](https://example.com/budget)" in enriched


@patch("rag_news.workers.question_worker.RedisStateStore")
@patch("rag_news.workers.question_worker.LLMClient")
@patch("rag_news.workers.question_worker.CongressAPI")
@patch("rag_news.workers.question_worker.KafkaProducer")
@patch("rag_news.workers.question_worker.KafkaConsumer")
def test_question_worker_dispatches_articles(
    mock_consumer,
    mock_producer,
    mock_api,
    mock_llm,
    mock_state,
    patch_redis,
):
    message = MagicMock()
    message.value = {
        "bill_id": "hr1",
        "bill_type": "hr",
        "congress": 119,
        "question_id": 1,
    }
    mock_consumer.return_value.__iter__.return_value = [message]
    mock_api.return_value.fetch_bill_bundle.return_value = {"bill": {"title": "Test"}}
    mock_api.return_value.query_context.return_value = ["context"]
    mock_llm.return_value.answer_question.return_value = "answer"
    mock_state_instance = mock_state.return_value
    mock_state_instance.all_questions_answered.return_value = True
    mock_state_instance.fetch_answers.return_value = {1: "answer"}

    worker = QuestionWorker()
    worker.run()

    mock_state_instance.record_answer.assert_called_once()
    mock_producer.return_value.send.assert_called_once()


@patch("rag_news.workers.link_worker.RedisStateStore")
@patch("rag_news.workers.link_worker.requests.head")
@patch("rag_news.workers.link_worker.KafkaConsumer")
def test_link_worker_writes_results(mock_consumer, mock_head, mock_state_store):
    mock_head.return_value.status_code = 200
    message = MagicMock()
    message.value = {
        "bill_id": "hr1",
        "article_text": "See http://example.com",
        "bill_payload": {
            "bill": {
                "title": "Test Bill",
                "sponsors": [{"bioguideId": "X001"}],
                "committees": {"committees": {"items": [{"systemCode": "HSBU"}]}},
            }
        },
    }
    mock_consumer.return_value.__iter__.return_value = [message]

    worker = LinkCheckWorker()
    worker.run()

    mock_state_store.return_value.append_article_bundle.assert_called_once()
    saved_payload = mock_state_store.return_value.append_article_bundle.call_args.args[0]
    assert "invalid_links" not in saved_payload

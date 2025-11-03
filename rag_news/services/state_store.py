"""Redis persistence for question progress and outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .redis_client import get_client
from .. import config


class RedisStateStore:
    """Stores answers, statuses, and progress in Redis."""

    def __init__(self) -> None:
        self._redis = get_client()

    # ------------------------------------------------------------------
    # Low-level helpers

    def _question_key(self, bill_id: str, question_id: int) -> str:
        return f"bill:{bill_id.lower()}:q:{question_id}"

    def _bill_answers_key(self, bill_id: str) -> str:
        return f"bill:{bill_id.lower()}:answers"

    def _bill_status_key(self, bill_id: str) -> str:
        return f"bill:{bill_id.lower()}:status"

    # ------------------------------------------------------------------

    def record_answer(
        self,
        bill_id: str,
        question_id: int,
        question: str,
        answer: str,
    ) -> None:
        entry = {
            "bill_id": bill_id,
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._redis.hset(self._question_key(bill_id, question_id), mapping=entry)
        self._redis.sadd(self._bill_status_key(bill_id), question_id)
        self._redis.hset(self._bill_answers_key(bill_id), question_id, answer)

    def has_answer(self, bill_id: str, question_id: int) -> bool:
        return self._redis.hexists(self._bill_answers_key(bill_id), question_id)

    def fetch_answers(self, bill_id: str) -> Dict[int, str]:
        data = self._redis.hgetall(self._bill_answers_key(bill_id))
        return {int(k): v for k, v in data.items()}

    def all_questions_answered(self, bill_id: str) -> bool:
        answered = self._redis.scard(self._bill_status_key(bill_id))
        return answered >= len(config.QUESTIONS)

    def reset_bill(self, bill_id: str) -> None:
        for q_id in config.QUESTIONS.keys():
            self._redis.delete(self._question_key(bill_id, q_id))
        self._redis.delete(self._bill_answers_key(bill_id))
        self._redis.delete(self._bill_status_key(bill_id))

    # ------------------------------------------------------------------
    # Output helpers

    def append_answer_bundle(self, bill_id: str, answers: Dict[int, str]) -> None:
        bundle = {"bill_id": bill_id, "answers": answers}
        self._upsert_json_record(config.OUTPUT_ANSWERS_FILE, "bill_id", bundle)

    def append_article_bundle(self, payload: dict) -> None:
        self._upsert_json_record(config.OUTPUT_ARTICLES_FILE, "bill_id", payload)

    @staticmethod
    def _upsert_json_record(path, key_field: str, payload: dict) -> None:
        try:
            with path.open("r", encoding="utf-8") as fh:
                records = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            records = []

        updated = False
        for idx, existing in enumerate(records):
            if isinstance(existing, dict) and existing.get(key_field) == payload.get(key_field):
                records[idx] = payload
                updated = True
                break

        if not updated:
            records.append(payload)

        with path.open("w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=4)

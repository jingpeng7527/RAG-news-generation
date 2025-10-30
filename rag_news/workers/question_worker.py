"""Worker responsible for question answering using cached RAG context."""

from __future__ import annotations

import json
import threading
from typing import Dict

from kafka import KafkaConsumer, KafkaProducer

from .. import config
from ..services import CongressAPI, LLMClient, RedisStateStore


class QuestionWorker(threading.Thread):
    """Consumes query tasks, answers questions, and persists results."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.consumer = KafkaConsumer(
            config.KAFKA_TOPICS["query"],
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="rag-question-workers",
            enable_auto_commit=True,
            auto_offset_reset="earliest",
        )
        self.producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.api = CongressAPI()
        self.llm = LLMClient()
        self.state = RedisStateStore()

    def run(self) -> None:  # pragma: no cover - infinite loop
        print("[question] worker ready")
        for message in self.consumer:
            payload: Dict = message.value
            bill_id = payload["bill_id"]
            bill_type = payload["bill_type"]
            congress = payload["congress"]
            question_id = payload["question_id"]
            question = config.QUESTIONS.get(question_id, "")

            print(f"[question] bill {bill_id.upper()} q{question_id} queued")
            context = self.api.query_context(bill_id, question)
            answer = self.llm.answer_question(question, context)
            print(f"[question] bill {bill_id.upper()} q{question_id} answered")

            self.state.record_answer(bill_id, question_id, question, answer)

            if self.state.all_questions_answered(bill_id):
                answers = self.state.fetch_answers(bill_id)
                self.state.append_answer_bundle(bill_id, answers)
                print(f"[question] all answers ready for {bill_id.upper()}, dispatching article task")
                self.producer.send(
                    config.KAFKA_TOPICS["article"],
                    {
                        "bill_id": bill_id,
                        "bill_type": bill_type,
                        "congress": congress,
                    },
                )

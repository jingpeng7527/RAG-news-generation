"""Worker that assembles Markdown articles from persisted answers."""

from __future__ import annotations

import json
import threading
from typing import Dict

from kafka import KafkaConsumer, KafkaProducer

from .. import config
from ..services import CongressAPI, LLMClient, RedisStateStore, add_hyperlinks


class ArticleWorker(threading.Thread):
    """Generates articles once all questions for a bill are answered."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.consumer = KafkaConsumer(
            config.KAFKA_TOPICS["article"],
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="rag-article-workers",
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
        print("[article] worker ready")
        for message in self.consumer:
            payload: Dict = message.value
            bill_id = payload["bill_id"]
            bill_type = payload["bill_type"]
            congress = payload["congress"]

            print(f"[article] drafting summary for {bill_id.upper()}")
            answers = self.state.fetch_answers(bill_id)
            bundle = self.api.fetch_bill_bundle(bill_id, bill_type, congress)

            # Get context from vector store (semantic search)
            retrieval_prompt = f"news article about {bill_id.upper()} including statuses, committees, sponsors, and recent activity"
            context = self.api.query_context(bill_id, retrieval_prompt)
            
            # Model has limited context (2048 tokens), so limit additional chunks
            # Only add a few more if vector store didn't return much
            if len(context) < 5:
                from ..services.vector_store import build_context_chunks
                all_chunks = build_context_chunks(bundle)
                # Add only 3-5 more chunks to stay within token limits
                context = list(context) + all_chunks[:5]
            
            article = self.llm.draft_article(bill_id, answers, context)
            article = add_hyperlinks(article, bundle)
            print(f"[article] draft complete for {bill_id.upper()}, sending to link check")

            self.producer.send(
                config.KAFKA_TOPICS["link_check"],
                {
                    "bill_id": bill_id,
                    "article_text": article,
                    "bill_payload": bundle,
                },
            )

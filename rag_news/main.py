"""Entry point for the redesigned RAG News system."""

from __future__ import annotations

import json
import time

from kafka import KafkaProducer
from .metrics import metrics

from rag_news import config
from rag_news.services import kafka_utils
from rag_news.services.state_store import RedisStateStore
from rag_news.workers import ArticleWorker, LinkCheckWorker, QuestionWorker


def main() -> None:
    app_start_time = time.time()
    print("[boot] starting RAG News pipeline")
    _prepare_output_files()
    kafka_utils.ensure_topics()

    # Clear Redis state for all bills before starting
    state = RedisStateStore()
    print("[boot] clearing previous answers from Redis...")
    for bill in config.TARGET_BILLS:
        state.reset_bill(bill["id"])
        print(f"[boot] reset {bill['id'].upper()}")
    
    # Small delay to ensure reset completes before workers start consuming
    time.sleep(0.5)

    workers = _spawn_workers()
    # Small delay to let workers initialize
    time.sleep(0.5)
    _dispatch_initial_tasks()

    _wait_for_completion()

    for worker in workers:
        worker.join(timeout=1)

    end_time = time.time()
    total_time = end_time - app_start_time
    print(f"Total application processing time: {total_time:.2f} seconds.")
    print(f"  - Time spent on Congress.gov API calls: {metrics.congress_api_time:.2f} seconds.")
    print(f"  - Time spent on LLM API calls: {metrics.llm_api_time:.2f} seconds.")
    print(f"Output written to {config.OUTPUT_ARTICLES_FILE} and {config.OUTPUT_ANSWERS_FILE}")

def _prepare_output_files() -> None:
    for path in (config.OUTPUT_ARTICLES_FILE, config.OUTPUT_ANSWERS_FILE):
        if path.exists():
            path.unlink()
        path.touch()
        path.write_text("[]", encoding="utf-8")


def _spawn_workers() -> list:
    workers = []
    for _ in range(config.NUM_QUERY_WORKERS):
        worker = QuestionWorker()
        worker.start()
        workers.append(worker)

    for _ in range(config.NUM_ARTICLE_WORKERS):
        worker = ArticleWorker()
        worker.start()
        workers.append(worker)

    link_worker = LinkCheckWorker()
    link_worker.start()
    workers.append(link_worker)

    return workers


def _dispatch_initial_tasks() -> None:
    producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    for bill in config.TARGET_BILLS:
        for question_id in config.QUESTIONS:
            producer.send(
                config.KAFKA_TOPICS["query"],
                {
                    "bill_id": bill["id"],
                    "bill_type": bill["type"],
                    "congress": bill["congress"],
                    "question_id": question_id,
                },
            )
    producer.flush()


def _wait_for_completion() -> None:
    target_articles = len(config.TARGET_BILLS)
    while True:
        try:
            with config.OUTPUT_ARTICLES_FILE.open("r", encoding="utf-8") as fh:
                articles = json.load(fh)
            if len(articles) >= target_articles:
                print("[done] all articles generated")
                return
        except json.JSONDecodeError:
            pass
        time.sleep(1)


if __name__ == "__main__":
    main()

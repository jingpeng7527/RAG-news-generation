"""Kafka helpers for provisioning topics and creating producers."""

from __future__ import annotations

import json
import logging
import time

from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import NoBrokersAvailable

from .. import config


log = logging.getLogger(__name__)


def ensure_topics() -> None:
    topics = config.build_topic_config()
    desired = set(topics.values())

    while True:
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                client_id="rag-news-admin",
            )
            existing = set(admin.list_topics())
            missing = desired - existing
            if not missing:
                return
            new_topics = [
                NewTopic(name=topic, num_partitions=4 if topic in (topics["query"], topics["article"]) else 1, replication_factor=1)
                for topic in missing
            ]
            admin.create_topics(new_topics=new_topics, validate_only=False)
            return
        except NoBrokersAvailable:
            time.sleep(2)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Kafka topic creation failed: %s", exc)
            time.sleep(2)


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

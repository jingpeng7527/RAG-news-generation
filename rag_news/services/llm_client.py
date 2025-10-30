"""Client for interacting with the local Ollama LLM server."""

from __future__ import annotations

import threading
import logging
from typing import Iterable, List, Optional

import requests

from .. import config


class LLMClient:
    """Thread-safe wrapper around the Ollama HTTP APIs."""

    _instance: Optional["LLMClient"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "LLMClient":  # pragma: no cover - singleton plumbing
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._session = requests.Session()
        return cls._instance

    def __init__(self) -> None:
        self._base = config.LLM_API_BASE
        self._model = config.LLM_MODEL
        self._embedding_model = config.EMBEDDING_MODEL

    # ------------------------------------------------------------------
    # Public helpers

    def answer_question(self, question: str, context_snippets: Iterable[str]) -> str:
        context_block = "\n\n".join(context_snippets)
        prompt = (
            "You are an assistant that writes factual answers about US legislation.\n"
            "Use the supplied context to respond concisely and cite relevant facts.\n"
            "Context:\n"
            f"{context_block}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self._invoke_chat(prompt, temperature=0.2)

    def draft_article(self, bill_id: str, answers: dict[int, str], context_snippets: Iterable[str]) -> str:
        context_block = "\n\n".join(context_snippets)
        ordered_answers = "\n".join(
            f"Q{qid}: {answers.get(qid, 'Not answered')}" for qid in sorted(answers)
        )
        prompt = (
            f"You are a journalist summarising U.S. congressional bill {bill_id.upper()}.\n"
            "Use the provided answers and context to write an objective Markdown article.\n"
            "Include a short headline and separate paragraphs, avoid speculation.\n"
            "Answers:\n"
            f"{ordered_answers}\n\n"
            "Additional context:\n"
            f"{context_block}"
        )
        return self._invoke_chat(prompt, temperature=0.6)

    def embed_text(self, text: str) -> List[float]:
        payload = {"model": self._embedding_model, "input": text}
        response = self._session.post(
            f"{self._base}/embeddings",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            if "embedding" in data and isinstance(data["embedding"], list):
                return data["embedding"]
            data_field = data.get("data")
            if isinstance(data_field, list) and data_field:
                embedding = data_field[0].get("embedding")
                if isinstance(embedding, list):
                    return embedding

        logging.warning("Received invalid embedding payload: %s", data)
        return []

    def _invoke_chat(self, prompt: str, *, temperature: float) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = self._session.post(
            f"{self._base}/chat",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {}).get("content", "").strip()
        return message

    @property
    def _session(self) -> requests.Session:
        return self.__dict__["_session"]

    @_session.setter
    def _session(self, value: requests.Session) -> None:
        self.__dict__["_session"] = value

"""Client for interacting with the local Ollama LLM server."""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, List, Optional

import requests

from .. import config


log = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the local LLM server cannot satisfy a request."""


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
        context_block = self._format_context(context_snippets, max_total_chars=6000, max_snippet_chars=1200)
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
        context_block = self._format_context(context_snippets, max_total_chars=8000, max_snippet_chars=1600)
        ordered_answers = "\n".join(
            f"Q{qid}: {answers.get(qid, 'Not answered')}" for qid in sorted(answers)
        )
        prompt = (
            f"You are a journalist summarising U.S. congressional bill {bill_id.upper()}.\n"
            "Use the provided answers and context to write an objective Markdown article.\n"
            "Use Markdown for the final output:\n"
            "- Begin with a **bold headline** that summarises the bill in 12 words or fewer.\n"
            "- Use clear, factual paragraphs only (no bullet points, no lists).\n"
            "- Keep a neutral, professional tone similar to a news wire or government press report.\n"
            "Do NOT invent or assume information that isn’t in the input.\n"
            "If information is unclear or missing, omit it without speculation.\n"
            "Maintain separate paragraphs and avoid speculation.\n"
            "Answers:\n"
            f"{ordered_answers}\n\n"
            "Additional context:\n"
            f"{context_block}"
        )
        article = self._invoke_chat(prompt, temperature=0.6).strip()
        return self._ensure_headline(article, bill_id, answers)

    def embed_text(self, text: str) -> List[float]:
        payload = {"model": self._embedding_model, "input": text}
        try:
            response = self._session.post(
                f"{self._base}/embeddings",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=config.LLM_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:  # pragma: no cover - runtime failure
            raise LLMClientError(f"Failed to generate embedding: {exc}") from exc
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
        attempts = max(config.LLM_MAX_RETRIES, 0) + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._session.post(
                    f"{self._base}/chat",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=config.LLM_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                message = data.get("message", {}).get("content", "").strip()
                return message
            except requests.RequestException as exc:
                if attempt >= attempts:
                    raise LLMClientError(
                        f"Chat completion failed after {attempts} attempt(s): {exc}"
                    ) from exc
                backoff = config.LLM_RETRY_BACKOFF_SECONDS * attempt
                log.warning(
                    "LLM chat request failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt,
                    attempts,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        raise LLMClientError("Chat completion failed with no response from LLM")

    def _format_context(
        self,
        snippets: Iterable[str],
        *,
        max_total_chars: int,
        max_snippet_chars: int,
    ) -> str:
        selected: list[str] = []
        total = 0
        trimmed = False

        for raw in snippets:
            text = raw.strip()
            if not text:
                continue

            if len(text) > max_snippet_chars:
                text = text[:max_snippet_chars].rstrip()
                trimmed = True

            remaining = max_total_chars - total
            if remaining <= 0:
                trimmed = True
                break

            if len(text) > remaining:
                text = text[:remaining].rstrip()
                trimmed = True

            selected.append(text)
            total += len(text)

            if total >= max_total_chars:
                trimmed = True
                break

        if trimmed:
            log.info(
                "Context trimmed to %s characters across %s snippet(s)",
                total,
                len(selected),
            )
        return "\n\n".join(selected)

    def _ensure_headline(self, article: str, bill_id: str, answers: dict[int, str]) -> str:
        trimmed = article.lstrip()
        lines = trimmed.splitlines()
        if lines and lines[0].startswith("#"):
            # Ensure there is exactly one leading blank line after the headline.
            rest = "\n".join(lines[1:]).lstrip("\n")
            return "\n".join([lines[0].rstrip(), "", rest]).rstrip() if rest else lines[0].rstrip()

        headline = self._build_fallback_headline(bill_id, answers)
        if trimmed:
            body = trimmed
        else:
            body = ""
        combined = f"{headline}\n\n{body}".strip()
        return combined

    def _build_fallback_headline(self, bill_id: str, answers: dict[int, str]) -> str:
        primary_answer = answers.get(1, "").strip()
        if primary_answer:
            first_sentence = primary_answer.split(".")[0].strip()
            if first_sentence:
                return f"# {first_sentence}"
        return f"# Update on {bill_id.upper()}"

    @property
    def _session(self) -> requests.Session:
        return self.__dict__["_session"]

    @_session.setter
    def _session(self, value: requests.Session) -> None:
        self.__dict__["_session"] = value

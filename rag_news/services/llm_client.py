"""Client for interacting with the configured local LLM server."""

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
    """Thread-safe wrapper around the local LLM HTTP APIs."""

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
        self._provider = config.LLM_PROVIDER
        self._base = config.LLM_API_BASE.rstrip("/")
        self._model = config.LLM_MODEL
        self._embedding_model = config.EMBEDDING_MODEL
        if self._provider not in {"ollama", "lmstudio"}:
            raise ValueError(f"Unsupported LLM provider: {self._provider}")
        if self._provider == "lmstudio":
            self._chat_endpoint = f"{self._base}/chat/completions"
        else:
            self._chat_endpoint = f"{self._base}/chat"
        self._embeddings_endpoint = f"{self._base}/embeddings"

    # ------------------------------------------------------------------
    # Public helpers

    def answer_question(self, question: str, context_snippets: Iterable[str]) -> str:
        # Model has 2048 token limit (~8000 chars total including prompt), be conservative
        # Limit to 3000 chars for context (leaves room for prompt + response)
        context_block = self._format_context(context_snippets, max_total_chars=3000, max_snippet_chars=800)
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
        # Model has 2048 token limit (~8000 chars total including prompt + answers + response)
        # Answers already take space, limit context to 2000 chars (leaves room for prompt + answers + response)
        context_block = self._format_context(context_snippets, max_total_chars=2000, max_snippet_chars=600)
        ordered_answers = "\n".join(
            f"Q{qid}: {answers.get(qid, 'Not answered')}" for qid in sorted(answers)
        )
        prompt = (
            f"You are a journalist summarising U.S. congressional bill {bill_id.upper()}.\n"
            "Use the provided answers and context to write an objective Markdown article.\n"
            "Strict output format:\n"
            "1. First line: **Bold headline** of 12 words or fewer summarising the bill.\n"
            "2. Subsequent lines: Plain paragraphs (no bullet points) conveying only facts from Answers/Context.\n"
            "Do NOT invent or amplify information beyond the supplied material.\n"
            "Maintain a neutral, wire-report tone and attribute statements to provided sources when possible.\n"
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
                self._embeddings_endpoint,
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
        payload = self._build_chat_payload(prompt, temperature)
        attempts = max(config.LLM_MAX_RETRIES, 0) + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._session.post(
                    self._chat_endpoint,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=config.LLM_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                message = self._extract_chat_message(data)
                if not message:
                    raise LLMClientError("LLM response did not include content.")
                return message
            except requests.RequestException as exc:
                # Log more details for 400 errors to help debug
                if hasattr(exc, 'response') and exc.response is not None:
                    try:
                        error_detail = exc.response.text[:500]
                        log.warning(
                            "LLM chat request failed (attempt %s/%s): %s - Response: %s",
                            attempt,
                            attempts,
                            exc,
                            error_detail,
                        )
                    except Exception:
                        pass
                
                if attempt >= attempts:
                    error_msg = f"Chat completion failed after {attempts} attempt(s): {exc}"
                    if hasattr(exc, 'response') and exc.response is not None:
                        try:
                            error_detail = exc.response.text[:500]
                            error_msg += f" - Response: {error_detail}"
                        except Exception:
                            pass
                    raise LLMClientError(error_msg) from exc
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

    def _build_chat_payload(self, prompt: str, temperature: float) -> dict:
        if self._provider == "lmstudio":
            return {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 1024,  # Reduced to fit within model's 2048 token context limit
                "stream": False,
            }
        return {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        }

    def _extract_chat_message(self, data: dict) -> str:
        if self._provider == "lmstudio":
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content.strip()
            return ""
        message = data.get("message", {})
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
        return ""

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
            headline_text = lines[0].lstrip("# ").strip()
            lines[0] = f"**{headline_text}**" if headline_text else "**Update**"
        if lines and lines[0].startswith("**") and lines[0].rstrip().endswith("**"):
            rest = "\n".join(lines[1:]).lstrip("\n")
            formatted_headline = lines[0].strip()
            return "\n".join([formatted_headline, "", rest]).rstrip() if rest else formatted_headline

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
                return f"**{first_sentence[:60]}**"
        return f"**Update on {bill_id.upper()}**"

    @property
    def _session(self) -> requests.Session:
        return self.__dict__["_session"]

    @_session.setter
    def _session(self, value: requests.Session) -> None:
        self.__dict__["_session"] = value

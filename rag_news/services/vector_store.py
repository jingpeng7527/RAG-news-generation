"""Lightweight FAISS-like vector storage using Chroma."""

from __future__ import annotations

import json
from typing import Iterable, List
from uuid import uuid4

from .. import config
from . import llm_client

try:  # pragma: no cover - import guard for optional dependency
    import chromadb  # type: ignore
    from chromadb.utils import embedding_functions  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    chromadb = None
    embedding_functions = None


if embedding_functions is not None:
    BaseEmbeddingFunction = embedding_functions.EmbeddingFunction
else:  # pragma: no cover - fallback for missing dependency
    class BaseEmbeddingFunction:  # type: ignore[misc]
        def __call__(self, texts):
            raise RuntimeError("chromadb is not installed; vector store is unavailable.")


class OllamaEmbeddingFunction(BaseEmbeddingFunction):
    """Embedding function that delegates to the local Ollama embedding endpoint."""

    def __init__(self) -> None:
        self._client = llm_client.LLMClient()
        self._last_dim: int | None = None

    def __call__(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        embeddings: List[List[float]] = []
        for text in texts:
            vector = self._client.embed_text(text)
            if not vector:
                vector = [0.0] * (self._last_dim or 1)
            else:
                self._last_dim = len(vector)
            embeddings.append(vector)
        return embeddings


class BillVectorStore:
    """Wrapper around a persistent Chroma collection."""

    def __init__(self) -> None:
        if chromadb is None:
            raise RuntimeError(
                "chromadb is required for the vector store. Install it via `pip install chromadb`."
            )
        config.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_PATH))
        self._collection = self._client.get_or_create_collection(
            name="bill-knowledge",
            embedding_function=OllamaEmbeddingFunction(),
        )

    def upsert_bill(self, bill_id: str, contexts: Iterable[str]) -> None:
        bill_id = bill_id.lower()
        self._collection.delete(where={"bill_id": bill_id})

        documents: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []
        for chunk in contexts:
            text = chunk.strip()
            if not text:
                continue
            documents.append(text)
            metadatas.append({"bill_id": bill_id})
            ids.append(f"{bill_id}-{uuid4()}" )

        if not documents:
            return

        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, bill_id: str, prompt: str, top_k: int | None = None) -> List[str]:
        if top_k is None:
            top_k = config.VECTOR_TOP_K

        result = self._collection.query(
            query_texts=[prompt],
            n_results=top_k,
            where={"bill_id": bill_id.lower()},
        )
        documents = result.get("documents", [[]])[0]
        return documents


def build_context_chunks(bill_payload: dict) -> List[str]:
    """Extract lightweight textual chunks from the Congress.gov payload."""

    bill = bill_payload.get("bill", {})
    chunks: List[str] = []

    title = bill.get("title")
    if title:
        chunks.append(f"Title: {title}")

    summary_items = _extract_items(bill.get("summaries"))
    for item in summary_items:
        text = item.get("text")
        if text:
            chunks.append(f"Summary: {text}")


    committee_items = _extract_items(bill.get("committees"), default_key="committees")
    for committee in committee_items:
        name = committee.get("name") or committee.get("committeeName")
        if name:
            chunks.append(f"Committee: {name}")

    sponsors = bill.get("sponsors", [])
    for sponsor in sponsors:
        full = sponsor.get("fullName")
        role = sponsor.get("party")
        if full:
            party = f" ({role})" if role else ""
            chunks.append(f"Sponsor: {full}{party}")

    cosponsors = _extract_items(bill.get("cosponsors"))
    cosponsor_names = [item.get("fullName") for item in cosponsors if item.get("fullName")]
    if cosponsor_names:
        chunks.append("Cosponsors: " + ", ".join(cosponsor_names[:20]))

    amendments = _extract_items(bill.get("amendments"))
    for amendment in amendments[:5]:
        chunks.append("Amendment: " + json.dumps(amendment, ensure_ascii=False))

    actions = bill.get("actions")
    action_items = _extract_items(actions)
    for action in action_items[:5]:
        desc = action.get("text")
        if desc:
            chunks.append("Recent action: " + desc)

    if not chunks:
        chunks.append(json.dumps(bill_payload, ensure_ascii=False)[:2000])

    return chunks


def _extract_items(section, *, default_key: str = "items", _depth: int = 0) -> List[dict]:
    """Return a list of item dictionaries regardless of API shape."""

    if _depth > 5:  # pragma: no cover - safety against unexpected recursion
        return []

    if isinstance(section, list):
        return [item for item in section if isinstance(item, dict)]

    if isinstance(section, dict):
        keys_to_try = [default_key]
        if "items" not in keys_to_try:
            keys_to_try.append("items")

        for key in keys_to_try:
            if key in section:
                items = _extract_items(section[key], default_key=default_key, _depth=_depth + 1)
                if items:
                    return items

        for value in section.values():
            if isinstance(value, (list, dict)):
                items = _extract_items(value, default_key=default_key, _depth=_depth + 1)
                if items:
                    return items

        # Fallback to treating the dict itself as a single item if it contains scalar data
        if section and all(not isinstance(v, (list, dict)) for v in section.values()):
            return [section]

    return []

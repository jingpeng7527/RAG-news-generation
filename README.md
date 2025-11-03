# RAG News Pipeline
Made by Jing Peng

Retrieval-augmented workflow that turns Congress.gov data into Markdown briefs using Kafka, Redis, Chroma, and LM Studio—all packaged in Docker Compose.

## i. Setup Instructions

1. Install prerequisites:
```bash
# Install and run LM Studio
# Download from https://lmstudio.ai/
# Start LM Studio and load a model (e.g., Qwen3-4B-Thinking)
# Ensure LM Studio server is running on http://127.0.0.1:1234
pip install -r requirements.txt
```

2. Add a `.env` file (not committed):
```bash
CONGRESS_API_KEY=your_api_key
LLM_PROVIDER=lmstudio
# For local development (outside Docker):
LMSTUDIO_BASE_URL=http://127.0.0.1:1234
# For Docker Compose (use this):
# LMSTUDIO_BASE_URL=http://host.docker.internal:1234
LLM_MODEL=qwen/qwen3-4b-thinking-2507
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m
REDIS_URL=redis://redis:6379/0
```

## ii. Architecture Overview

The pipeline uses a distributed architecture with three worker types orchestrated via Kafka message queues:

**Data Flow:**
1. **QuestionWorker** processes individual questions about bills using Congress.gov API data and vector search
2. **ArticleWorker** synthesizes answers into coherent news articles using LM Studio LLM
3. **LinkCheckWorker** validates all hyperlinks and writes final JSON output

**Key Components:**
- **Workers** (`rag_news/workers/`):
  - `question_worker.py` – Answers questions about bills, caches results in Redis
  - `article_worker.py` – Generates AI articles with official hyperlinks
  - `link_worker.py` – Validates URLs and writes final output

- **Services** (`rag_news/services/`):
  - `congress_api.py` – Congress.gov API client with caching
  - `vector_store.py` – Chroma vector database for semantic search
  - `llm_client.py` – LM Studio integration for text generation
  - `redis_client.py` – Caching and state management
  - `kafka_utils.py` – Message queue administration

- **Infrastructure:**
  - **Kafka** – Message orchestration between workers
  - **Redis** – High-speed caching and state persistence
  - **Chroma** – Vector database for semantic search
  - **LM Studio** – Local LLM serving

## iii. How to Run the Pipeline

```bash
docker-compose down --remove-orphans
docker-compose up --build --exit-code-from app --remove-orphans
```

To run tests (no Docker needed):
```bash
pytest
```

## iv. Example Output

```json
{
  "bill_id": "hr5401",
  "bill_title": "Pay Our Troops Act of 2026",
  "sponsor_bioguide_id": "K000399",
  "bill_committee_ids": ["hsap00"],
  "article_content": "## Pay Our Troops Act of 2026 Moves Forward..."
}
```

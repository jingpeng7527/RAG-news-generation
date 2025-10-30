# RAG News Pipeline

Retrieval-augmented workflow that turns Congress.gov data into Markdown briefs using Kafka, Redis, Chroma, and Ollama—all packaged in Docker Compose.

## i. Setup Instructions

1. Install prerequisites:
```bash
ollama pull gemma:2b
ollama pull nomic-embed-text
ollama serve --address 0.0.0.0
pip install -r requirements.txt
```

2. Add a `.env` file (not committed):
```bash
CONGRESS_API_KEY=your_api_key
LLM_HOST=host.docker.internal
LLM_PORT=11434
REDIS_URL=redis://redis:6379/0
```

## ii. Architecture Overview

The pipeline uses a distributed architecture with three worker types orchestrated via Kafka message queues:

**Data Flow:**
1. **QuestionWorker** processes individual questions about bills using Congress.gov API data and vector search
2. **ArticleWorker** synthesizes answers into coherent news articles using Ollama LLM
3. **LinkCheckWorker** validates all hyperlinks and writes final JSON output

**Key Components:**
- **Workers** (`rag_news/workers/`):
  - `question_worker.py` – Answers questions about bills, caches results in Redis
  - `article_worker.py` – Generates AI articles with official hyperlinks
  - `link_worker.py` – Validates URLs and writes final output

- **Services** (`rag_news/services/`):
  - `congress_api.py` – Congress.gov API client with caching
  - `vector_store.py` – Chroma vector database for semantic search
  - `llm_client.py` – Ollama integration for text generation
  - `redis_client.py` – Caching and state management
  - `kafka_utils.py` – Message queue administration

- **Infrastructure:**
  - **Kafka** – Message orchestration between workers
  - **Redis** – High-speed caching and state persistence
  - **Chroma** – Vector database for semantic search
  - **Ollama** – Local LLM serving

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

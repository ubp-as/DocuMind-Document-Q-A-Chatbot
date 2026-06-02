# 🔍 DocLens — Document Q&A Chatbot

A production-grade RAG (Retrieval-Augmented Generation) chatbot powered by **LangChain**, **OpenAI / Anthropic**, **ChromaDB**, and **FastAPI**. Upload PDFs or text files and ask questions in plain English — DocLens retrieves the most relevant passages and synthesizes accurate, cited answers.

## ✨ Features

- **RAG pipeline** — Answers grounded in your documents, not hallucinations
- **Multi-document support** — Upload and query across multiple PDFs or text files
- **Citation tracking** — Every answer links back to its source passage(s)
- **LLM-agnostic** — Swap between OpenAI GPT-4o and Anthropic Claude via a single env flag
- **RESTful API** — FastAPI backend with full OpenAPI docs at `/docs`
- **Query logging** — All queries + responses stored in PostgreSQL for analytics
- **Vector search** — ChromaDB with persistent storage and cosine similarity
- **Streaming responses** — Server-Sent Events for real-time token streaming
- **Docker Compose** — One command to run everything

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        User Query                        │
└─────────────────────┬───────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   FastAPI App   │  ← REST API layer
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
┌────────▼──────┐ ┌────▼────┐ ┌────▼────────┐
│  RAG Service  │ │  DB     │ │  Ingestion  │
│  (LangChain)  │ │  Logger │ │  Service    │
└────────┬──────┘ └────┬────┘ └────┬────────┘
         │             │           │
┌────────▼──────┐ ┌────▼──────┐ ┌─▼──────────┐
│   ChromaDB    │ │PostgreSQL │ │ Doc Loader │
│ (Vector Store)│ │(Query Log)│ │ + Chunker  │
└────────┬──────┘ └───────────┘ └────────────┘
         │
┌────────▼──────┐
│  OpenAI /     │
│  Anthropic    │
│  (LLM + Emb.) │
└───────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI or Anthropic API key

### 1. Clone & configure

```bash
git clone https://github.com/YOUR_USERNAME/doclens.git
cd doclens
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

API live at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`

### 3. Run locally (dev mode)

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start PostgreSQL via Docker only
docker compose up postgres -d

# Run migrations
alembic upgrade head

# Start the API
uvicorn app.main:app --reload
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload a PDF or .txt file |
| `GET` | `/api/v1/documents/` | List all indexed documents |
| `DELETE` | `/api/v1/documents/{id}` | Remove a document |
| `POST` | `/api/v1/chat/query` | Ask a question (JSON response) |
| `POST` | `/api/v1/chat/stream` | Ask a question (SSE streaming) |
| `GET` | `/api/v1/chat/history` | Retrieve query history |
| `GET` | `/api/v1/health` | Health check |

### Example: Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@my_paper.pdf"
```

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 4}'
```

**Response:**
```json
{
  "answer": "The main findings show that...",
  "sources": [
    {
      "document": "my_paper.pdf",
      "page": 3,
      "excerpt": "Our experiments demonstrate..."
    }
  ],
  "model": "gpt-4o",
  "tokens_used": 842,
  "latency_ms": 1243
}
```

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v --cov=app
```

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `LLM_MODEL` | `gpt-4o` | OpenAI model name |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Anthropic model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `800` | Token chunk size for splitting |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store path |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins (comma-separated) |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max file upload size |

## 🗂️ Project Structure

```
doclens/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI route handlers
│   │   │   ├── chat.py       # Chat query & streaming endpoints
│   │   │   └── documents.py  # Document upload/list/delete endpoints
│   │   ├── core/             # Config, RAG engine, vector store
│   │   │   ├── config.py
│   │   │   ├── rag_engine.py
│   │   │   └── vector_store.py
│   │   ├── db/               # SQLAlchemy models & session
│   │   │   └── models.py
│   │   ├── services/         # Business logic
│   │   │   ├── ingestion.py
│   │   │   └── query_logger.py
│   │   └── main.py           # App entrypoint & lifespan
│   ├── tests/
│   │   └── test_api.py
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   └── bulk_ingest.py        # CLI tool to bulk-upload a directory
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Orchestration | LangChain |
| LLM APIs | OpenAI GPT-4o / Anthropic Claude |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (persistent) |
| API Framework | FastAPI + Uvicorn |
| Database | PostgreSQL + SQLAlchemy + Alembic |
| Containerization | Docker + Docker Compose |
| Testing | Pytest + HTTPX |

## 📄 License

MIT

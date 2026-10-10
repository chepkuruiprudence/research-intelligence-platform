# Research Intelligence Platform — Hybrid RAG + AI Agents

> A learn-as-you-build guide. You will ingest research papers, answer questions with **verifiable citations**, compare findings across studies, detect contradictions, and surface research gaps — and you will measure how good the system actually is.

**Stack:** Python · FastAPI · PostgreSQL + pgvector · OpenSearch · GROBID · Hugging Face Transformers / PyTorch · LangGraph · Next.js + Material UI · Docker Compose

---

## Table of Contents

1. [How to use this guide](#1-how-to-use-this-guide)
2. [Concept primer (read once)](#2-concept-primer-read-once)
3. [Architecture](#3-architecture)
4. [Roadmap and technologies per phase](#4-roadmap-and-technologies-per-phase)
5. [Final folder structure](#5-final-folder-structure)
6. [Phase 0 — Tools and repository](#phase-0--tools-and-repository)
7. [Phase 1 — Docker infrastructure](#phase-1--docker-infrastructure)
8. [Phase 2 — Backend skeleton (FastAPI)](#phase-2--backend-skeleton-fastapi)
9. [Phase 3 — Frontend skeleton (Next.js + MUI)](#phase-3--frontend-skeleton-nextjs--mui)
10. [Phase 4 — Gathering data](#phase-4--gathering-data)
11. [Phase 5 — Parsing PDFs and chunking](#phase-5--parsing-pdfs-and-chunking)
12. [Phase 6 — Embeddings and vector search](#phase-6--embeddings-and-vector-search)
13. [Phase 7 — BM25, hybrid retrieval, reranking](#phase-7--bm25-hybrid-retrieval-reranking)
14. [Phase 8 — Answers with citations and citation checking](#phase-8--answers-with-citations-and-citation-checking)
15. [Phase 9 — Knowledge graph and graph retrieval](#phase-9--knowledge-graph-and-graph-retrieval)
16. [Phase 10 — Multi-step agent (LangGraph)](#phase-10--multi-step-agent-langgraph)
17. [Phase 11 — Evaluation dashboard](#phase-11--evaluation-dashboard)
18. [Phase 12 — Frontend pages](#phase-12--frontend-pages)
19. [Phase 13 — Testing, CI, hardening](#phase-13--testing-ci-hardening)
20. [Endpoint catalogue](#endpoint-catalogue)
21. [Database schema](#database-schema)
22. [Troubleshooting](#troubleshooting)
23. [Stretch goals and resources](#stretch-goals-and-resources)

---

## 1. How to use this guide

Every phase follows the same loop. Do not skip the last two steps; that is where learning happens.

| Step | What you do |
|------|-------------|
| **Read** | Understand *what* and *why* before typing. |
| **Build** | Create the files shown. Type them, don't paste blindly. |
| **Verify** | Run the "Verify" commands. If it doesn't work, fix it before moving on. |
| **Break it** | Deliberately change something (wrong password, empty query) and observe the error. |
| **Log it** | Write 3–5 lines in `LEARNING_LOG.md`: what you learned, what confused you, what you'd change. |

**Commit after every phase** (`git commit -m "phase 3: frontend skeleton"`). Your git history becomes proof of your learning path.

**Rules for the code** (they keep the project maintainable):

1. **One endpoint = one file.** `papers_list.py`, `papers_get.py`, never one giant `papers.py`.
2. **Layers do one job each:**
   - `endpoints/` (controller): receives HTTP, validates input, calls a service, returns output. No business logic.
   - `services/`: business logic. One file per use-case.
   - `repositories/`: all database / OpenSearch queries. Nothing else touches SQL.
   - `schemas/`: Pydantic request/response shapes.
   - `models/`: SQLAlchemy tables.
3. **Small files, small functions.** If a function needs a scroll bar, split it.
4. **No secrets in code.** Everything configurable lives in `.env`.

---

## 2. Concept primer (read once)

| Term | Plain-English meaning | Where used |
|------|----------------------|------------|
| **RAG** (Retrieval-Augmented Generation) | Instead of asking an LLM to answer from memory, first *retrieve* relevant passages from your documents, then ask the LLM to answer *using only those passages*. | Phases 6–8 |
| **Chunk** | A passage of a paper (~300 words) that is stored and retrieved as one unit. | Phase 5 |
| **Embedding** | A list of numbers (a vector) representing the *meaning* of text. Similar meanings → nearby vectors. | Phase 6 |
| **Vector search** | Find chunks whose embeddings are closest to the question's embedding. Good at meaning ("car" ≈ "automobile"). | Phase 6 |
| **BM25** | Classic keyword ranking (TF-IDF family). Good at exact terms, names, acronyms, numbers. | Phase 7 |
| **Hybrid retrieval** | Run BM25 *and* vector search, then merge. Each covers the other's blind spot. | Phase 7 |
| **RRF** (Reciprocal Rank Fusion) | Merge ranked lists using ranks only: `score = Σ 1/(60 + rank)`. No score normalisation needed. | Phase 7 |
| **Reranker (cross-encoder)** | A slower, smarter model that reads *(question, chunk)* together and re-scores the top ~50 candidates. | Phase 7 |
| **Faithfulness** | Is every claim in the answer supported by the retrieved text? (anti-hallucination metric) | Phases 8, 11 |
| **Knowledge graph** | Nodes (papers, authors, methods, datasets) and edges (CITES, USES_METHOD, EVALUATED_ON). Lets you answer "which methods were tested on dataset X?" | Phase 9 |
| **Agent** | An LLM-driven program that plans, calls tools (search, graph, verifier), checks its own work, and loops if needed. | Phase 10 |
| **LangGraph** | A library to define an agent as a *state machine* (nodes + conditional edges) so its behaviour is explicit and debuggable. | Phase 10 |

---

## 3. Architecture

```
                         ┌──────────────────────────────┐
                         │  Next.js + MUI (port 3000)   │
                         │  Papers · Search · Ask ·     │
                         │  Agent · Graph · Evaluation  │
                         └──────────────┬───────────────┘
                                        │ REST + SSE
                         ┌──────────────▼───────────────┐
                         │   FastAPI backend (8000)     │
                         │ endpoints → services → repos │
                         └─┬─────────┬──────────┬───────┘
          ingestion        │         │          │   agent (LangGraph)
 ┌─────────────────────┐   │         │          │  ┌──────────────────────┐
 │ collectors (arXiv…) │   │         │          │  │ plan → retrieve →    │
 │ GROBID / PyMuPDF    │───┘         │          │  │ grade → write →      │
 │ chunker + embedder  │             │          │  │ verify → gaps        │
 └─────────────────────┘             │          │  └──────────────────────┘
                                     │          │
        ┌──────────────┐   ┌─────────▼──┐  ┌────▼─────────┐   ┌──────────┐
        │ PostgreSQL   │   │ OpenSearch │  │ GROBID (8070)│   │ LLM API  │
        │ + pgvector   │   │ (BM25)     │  │ PDF → TEI XML│   │ or Ollama│
        │ papers,chunks│   │ 9200       │  └──────────────┘   └──────────┘
        │ graph, evals │   └────────────┘
        └──────────────┘
```

**Data flow in one sentence:** collect paper metadata + PDFs → parse into sections/references → chunk → embed → store in Postgres (vectors) and OpenSearch (keywords) → at question time retrieve, rerank, generate with citations, verify citations, and log metrics.

---

## 4. Roadmap and technologies per phase

| Phase | Goal | Technologies | You will learn |
|-------|------|--------------|----------------|
| 0 | Tools + repo | Git, Docker Desktop, VS Code, Python 3.11, Node 20 | Project hygiene, `.env`, `.gitignore` |
| 1 | Run infrastructure | Docker, Docker Compose, PostgreSQL 16, pgvector | Containers, volumes, health checks |
| 2 | Backend skeleton | FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2, Alembic, psycopg | REST design, layered architecture, migrations |
| 3 | Frontend skeleton | Next.js (App Router), TypeScript, Material UI, Emotion | Server vs client components, theming, API clients |
| 4 | Gather data | arXiv API, OpenAlex, Semantic Scholar, httpx, feedparser | Rate limits, licensing, idempotent ingestion |
| 5 | Parse + chunk | GROBID, PyMuPDF, pdfplumber, lxml | Document structure, chunking trade-offs |
| 6 | Embeddings + vector search | sentence-transformers, PyTorch, pgvector (HNSW) | Embeddings, cosine distance, ANN indexes |
| 7 | BM25 + hybrid + rerank | OpenSearch, opensearch-py, RRF, cross-encoder | Why hybrid beats either alone |
| 8 | Grounded answers | Anthropic API (or Ollama), prompt design, regex, LLM judge | Citation formats, hallucination checks |
| 9 | Knowledge graph | LLM structured extraction, Postgres edges, recursive CTE | Graph modelling, multi-hop retrieval |
| 10 | Agent | LangGraph, SSE (Server-Sent Events) | State machines, tool use, self-verification |
| 11 | Evaluation | Recall@k, MRR, nDCG, faithfulness, MUI X Charts | Measuring instead of guessing |
| 12 | UI pages | MUI components, React hooks | Building a research UI |
| 13 | Quality | pytest, ruff, GitHub Actions | Tests, CI |

---

## 5. Final folder structure

This is where you will end up. You create it gradually; don't make it all today.

```
research-intelligence-platform/
├── README.md
├── LEARNING_LOG.md
├── .env                      # secrets (never committed)
├── .env.example              # template (committed)
├── .gitignore
├── docker-compose.yml
├── infra/
│   └── postgres/init.sql     # enables pgvector
├── data/                     # mounted into containers (git-ignored)
│   ├── pdfs/
│   ├── parsed/
│   └── eval/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── alembic.ini
│   ├── alembic/versions/
│   ├── tests/
│   └── app/
│       ├── main.py
│       ├── core/             # config.py, logging.py
│       ├── db/               # base.py, session.py
│       ├── models/           # paper.py, chunk.py, author.py, entity.py, edge.py, job.py, agent_run.py, eval.py
│       ├── schemas/          # paper.py, search.py, ask.py, graph.py, agent.py, eval.py
│       ├── repositories/     # paper_repository.py, chunk_repository.py, vector_repository.py, bm25_repository.py, graph_repository.py ...
│       ├── services/         # list_papers.py, get_paper.py, collect_arxiv.py, ingest_paper.py, hybrid_search.py, answer_question.py ...
│       ├── ingestion/
│       │   ├── collectors/   # base.py, arxiv.py, openalex.py
│       │   ├── parsers/      # grobid_client.py, tei_parser.py, pymupdf_parser.py, table_parser.py
│       │   ├── chunking/     # section_chunker.py
│       │   └── embedding/    # embedder.py
│       ├── retrieval/        # rrf.py, reranker.py
│       ├── llm/              # base.py, anthropic_client.py, ollama_client.py, prompts/
│       ├── verification/     # citation_parser.py, citation_verifier.py
│       ├── graph_extraction/ # extractor.py, schemas.py
│       ├── agents/           # state.py, graph.py, nodes/
│       ├── evaluation/       # retrieval_metrics.py, generation_metrics.py, runner.py
│       └── api/
│           ├── deps.py
│           └── v1/
│               ├── router.py
│               └── endpoints/   # one file per endpoint
└── frontend/
    ├── Dockerfile
    ├── package.json
    └── src/
        ├── app/              # pages (App Router)
        ├── components/       # layout/, papers/, search/, ask/, agent/, graph/, evaluation/
        ├── lib/api/          # client.ts + one file per resource
        ├── theme/            # theme.ts
        └── types/
```

---

# Phase 0 — Tools and repository

### What and why
Before code, you need a reproducible workspace. Docker means "it runs on my machine" becomes "it runs on every machine".

### Technologies
- **Git** — version control
- **Docker Desktop** — runs containers (includes Docker Compose v2)
- **VS Code** — editor (extensions: Python, Docker, ESLint)
- **Python 3.11** and **Node.js 20 LTS** — only needed on your host for scaffolding and editor autocompletion; the app itself runs in containers

### Steps

```bash
# 1. Check tools
git --version
docker --version
docker compose version
node --version      # v20.x
python3 --version   # 3.11+

# 2. Create the repo
mkdir research-intelligence-platform && cd research-intelligence-platform
git init
mkdir -p backend frontend infra/postgres data/pdfs data/parsed data/eval
touch LEARNING_LOG.md
```

**`.gitignore`**

```gitignore
# secrets
.env

# data
data/
!data/.gitkeep

# python
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/

# node
node_modules/
.next/

# editor / OS
.vscode/
.DS_Store
```

**`.env.example`** (copy to `.env` and edit)

```env
# --- PostgreSQL ---
POSTGRES_USER=rip
POSTGRES_PASSWORD=change_me
POSTGRES_DB=rip
DATABASE_URL=postgresql+psycopg://rip:change_me@postgres:5432/rip

# --- Backend ---
CORS_ORIGINS=["http://localhost:3000"]

# --- Frontend ---
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

```bash
cp .env.example .env
```

> **Why `postgres` in the URL, not `localhost`?** Inside Docker Compose, each service is reachable by its *service name*. The backend container talks to `postgres:5432`.

### Verify
`docker run hello-world` prints a welcome message.

---

# Phase 1 — Docker infrastructure

### What and why
Start with only the database. Add services one phase at a time so you always know which piece broke.

### Technologies
- **PostgreSQL 16 with pgvector** (image `pgvector/pgvector:pg16`) — relational data *and* vector similarity in one database
- **Docker volumes** — keep data when containers restart
- **Health checks** — make dependent services wait until the DB is ready

### Files

**`infra/postgres/init.sql`** — runs once, the first time the volume is created

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**`docker-compose.yml`** (initial version)

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: rip-postgres
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./infra/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  pgdata:
```

### Verify

```bash
docker compose up -d postgres
docker compose ps                      # postgres should be "healthy"
docker compose exec postgres psql -U rip -d rip -c "\dx"
# You should see the extension named: vector
```

### Break it
Change `POSTGRES_PASSWORD` in `.env`, run `docker compose up -d` again, and try to connect. Why does the old password still work? (Answer: the volume already holds the initialised DB. Run `docker compose down -v` to wipe it.)

---

# Phase 2 — Backend skeleton (FastAPI)

### What and why
Build the layered structure **first** with one trivial feature. Every later feature copies this pattern.

### Technologies
| Tool | Purpose |
|------|---------|
| **FastAPI** | Web framework; auto-generates interactive docs at `/docs` |
| **Uvicorn** | ASGI server that runs FastAPI |
| **Pydantic v2 / pydantic-settings** | Validation + typed configuration from `.env` |
| **SQLAlchemy 2.0** | ORM (Python classes ↔ tables) |
| **psycopg 3** | PostgreSQL driver |
| **Alembic** | Database migrations (version control for your schema) |
| **pgvector (Python)** | Vector column type for SQLAlchemy |

### 2.1 Create the files

**`backend/requirements.txt`**

```txt
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic-settings>=2.4
sqlalchemy>=2.0
psycopg[binary]>=3.2
alembic>=1.13
pgvector>=0.3
python-multipart>=0.0.9
httpx>=0.27
```

**`backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Copy requirements first so Docker caches the install layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
```

**Add the backend to `docker-compose.yml`** (under `services:`)

```yaml
  backend:
    build: ./backend
    container_name: rip-backend
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/code          # live reload while you edit
      - ./data:/data             # PDFs and parsed files
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2.2 Core: configuration

**`backend/app/__init__.py`** — empty file. Create an empty `__init__.py` in **every** package folder (`core`, `db`, `models`, `schemas`, `repositories`, `services`, `api`, `api/v1`, `api/v1/endpoints`).

**`backend/app/core/config.py`**

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Research Intelligence Platform"
    api_prefix: str = "/api/v1"
    database_url: str
    cors_origins: list[str] = ["http://localhost:3000"]
    data_dir: str = "/data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

> `lru_cache` makes the settings load once and be reused.

### 2.3 Database layer

**`backend/app/db/base.py`**

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Parent of all ORM models."""
```

**`backend/app/db/session.py`**

```python
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: one DB session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 2.4 First model, schema, repository, service, endpoints

**`backend/app/models/paper.py`**

```python
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Paper(Base):
    __tablename__ = "papers"
    __table_args__ = (UniqueConstraint("source", "source_id", name="uq_paper_source"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(30))          # "arxiv", "upload", "openalex"
    source_id: Mapped[str] = mapped_column(String(100))      # e.g. arXiv id "2005.11401"
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(200), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pdf_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # metadata_only -> downloaded -> parsed -> indexed | failed
    status: Mapped[str] = mapped_column(String(20), default="metadata_only")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

> The `UniqueConstraint(source, source_id)` makes ingestion **idempotent**: collecting the same arXiv paper twice never creates a duplicate.

**`backend/app/models/__init__.py`** — import every model so Alembic sees them:

```python
from app.models.paper import Paper  # noqa: F401
```

**`backend/app/schemas/paper.py`**

```python
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PaperRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source: str
    source_id: str
    title: str
    abstract: str | None
    year: int | None
    status: str
    created_at: datetime


class PaperList(BaseModel):
    total: int
    items: list[PaperRead]
```

**`backend/app/repositories/paper_repository.py`**

```python
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.paper import Paper


def list_papers(db: Session, limit: int, offset: int) -> tuple[int, list[Paper]]:
    total = db.scalar(select(func.count()).select_from(Paper)) or 0
    stmt = select(Paper).order_by(Paper.created_at.desc()).limit(limit).offset(offset)
    return total, list(db.scalars(stmt))


def get_paper(db: Session, paper_id: uuid.UUID) -> Paper | None:
    return db.get(Paper, paper_id)
```

**`backend/app/services/list_papers.py`**

```python
from sqlalchemy.orm import Session

from app.repositories import paper_repository
from app.schemas.paper import PaperList, PaperRead


def list_papers(db: Session, limit: int = 20, offset: int = 0) -> PaperList:
    total, rows = paper_repository.list_papers(db, limit, offset)
    return PaperList(total=total, items=[PaperRead.model_validate(r) for r in rows])
```

**`backend/app/services/get_paper.py`**

```python
import uuid

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.repositories import paper_repository
from app.schemas.paper import PaperRead


def get_paper(db: Session, paper_id: uuid.UUID) -> PaperRead:
    paper = paper_repository.get_paper(db, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperRead.model_validate(paper)
```

**Endpoints — one file each**

`backend/app/api/v1/endpoints/health.py`

```python
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter()


@router.get("/health", tags=["system"])
def health(db: Session = Depends(get_db)) -> dict:
    db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "up"}
```

`backend/app/api/v1/endpoints/papers_list.py`

```python
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.paper import PaperList
from app.services import list_papers as svc

router = APIRouter()


@router.get("/papers", response_model=PaperList, tags=["papers"])
def list_papers(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> PaperList:
    return svc.list_papers(db, limit, offset)
```

`backend/app/api/v1/endpoints/papers_get.py`

```python
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.paper import PaperRead
from app.services import get_paper as svc

router = APIRouter()


@router.get("/papers/{paper_id}", response_model=PaperRead, tags=["papers"])
def get_paper(paper_id: uuid.UUID, db: Session = Depends(get_db)) -> PaperRead:
    return svc.get_paper(db, paper_id)
```

**Router that wires every endpoint file** — `backend/app/api/v1/router.py`

```python
from fastapi import APIRouter

from app.api.v1.endpoints import health, papers_get, papers_list

api_router = APIRouter()
for module in (health, papers_list, papers_get):
    api_router.include_router(module.router)
```

> From now on, adding an endpoint = create one file + add its module name to that tuple.

**`backend/app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_prefix)
```

### 2.5 Migrations with Alembic

```bash
docker compose up -d --build backend
docker compose exec backend alembic init alembic
```

Edit **`backend/alembic/env.py`**: find the line `target_metadata = None` and replace the top of the file's config section with:

```python
from app.core.config import get_settings
from app.db.base import Base
import app.models  # noqa: F401  (registers all models)

config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata
```

(The `config.set_main_option` line goes right after `config = context.config`.)

```bash
# Generate the first migration from your models, then apply it
docker compose exec backend alembic revision --autogenerate -m "create papers"
docker compose exec backend alembic upgrade head
```

> **Rule:** never edit tables by hand. Change the model → `revision --autogenerate` → `upgrade head`.

### Verify

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","database":"up"}

curl http://localhost:8000/api/v1/papers
# {"total":0,"items":[]}
```

Open **http://localhost:8000/docs** — you get a free interactive API explorer.

### Break it
Stop Postgres (`docker compose stop postgres`) and call `/health`. Read the error. This is what a real outage looks like.

**Commit:** `git add . && git commit -m "phase 2: backend skeleton"`

---
# Phase 3 — Frontend skeleton (Next.js + MUI)

### What and why
A UI that talks to your API from day one lets you *see* every later feature working.

### Technologies
| Tool | Purpose |
|------|---------|
| **Next.js (App Router)** | React framework: routing by folder, server + client components |
| **TypeScript** | Types catch bugs before runtime |
| **Material UI (MUI)** | Ready-made accessible components |
| **Emotion** | CSS-in-JS engine MUI uses |
| **@mui/material-nextjs** | Makes MUI styles work with Next's server rendering |
| **@mui/x-charts** | Charts for the evaluation dashboard (Phase 11) |

### 3.1 Scaffold

```bash
# from the repo root
npx create-next-app@latest frontend \
  --typescript --app --eslint --src-dir --no-tailwind --import-alias "@/*"

cd frontend
npm install @mui/material @emotion/react @emotion/styled @emotion/cache \
            @mui/material-nextjs @mui/icons-material @mui/x-charts
cd ..
```

> No Node on your host? Run the scaffold inside Docker:
> `docker run --rm -it -v "$PWD":/work -w /work node:20-alpine npx create-next-app@latest frontend --typescript --app --eslint --src-dir --no-tailwind --import-alias "@/*"`

### 3.2 Dockerfile and compose service

**`frontend/Dockerfile`** (development image)

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev"]
```

**Add to `docker-compose.yml`**

```yaml
  frontend:
    build: ./frontend
    container_name: rip-frontend
    env_file: .env
    depends_on:
      - backend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules        # keep container's node_modules
      - /app/.next
```

Add a `frontend/.dockerignore` containing `node_modules` and `.next`.

### 3.3 Theme

**`frontend/src/theme/theme.ts`**

```ts
"use client";
import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#1f4e8c" },
    secondary: { main: "#2e7d6b" },
    background: { default: "#f6f7fb" },
  },
  shape: { borderRadius: 10 },
  typography: { fontFamily: "Inter, Roboto, Helvetica, Arial, sans-serif" },
});
```

### 3.4 Layout

**`frontend/src/app/layout.tsx`**

```tsx
import type { Metadata } from "next";
import { AppRouterCacheProvider } from "@mui/material-nextjs/v15-appRouter";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { theme } from "@/theme/theme";
import AppShell from "@/components/layout/AppShell";

export const metadata: Metadata = { title: "Research Intelligence Platform" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AppRouterCacheProvider>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <AppShell>{children}</AppShell>
          </ThemeProvider>
        </AppRouterCacheProvider>
      </body>
    </html>
  );
}
```

> The import path suffix (`v15-appRouter`) must match your Next.js major version. If it errors, check the `@mui/material-nextjs` docs for the matching folder (`v13-appRouter`, `v14-appRouter`, `v15-appRouter`, ...).

**`frontend/src/components/layout/navItems.ts`** — one place to add pages

```ts
import ArticleIcon from "@mui/icons-material/Article";
import SearchIcon from "@mui/icons-material/Search";
import QuestionAnswerIcon from "@mui/icons-material/QuestionAnswer";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import HubIcon from "@mui/icons-material/Hub";
import QueryStatsIcon from "@mui/icons-material/QueryStats";

export const navItems = [
  { label: "Papers", href: "/papers", icon: ArticleIcon },
  { label: "Search", href: "/search", icon: SearchIcon },
  { label: "Ask", href: "/ask", icon: QuestionAnswerIcon },
  { label: "Agent", href: "/agent", icon: SmartToyIcon },
  { label: "Graph", href: "/graph", icon: HubIcon },
  { label: "Evaluation", href: "/evaluation", icon: QueryStatsIcon },
];
```

**`frontend/src/components/layout/Sidebar.tsx`**

```tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Drawer from "@mui/material/Drawer";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Toolbar from "@mui/material/Toolbar";
import { navItems } from "./navItems";

export const DRAWER_WIDTH = 220;

export default function Sidebar() {
  const pathname = usePathname();
  return (
    <Drawer
      variant="permanent"
      sx={{ width: DRAWER_WIDTH, "& .MuiDrawer-paper": { width: DRAWER_WIDTH } }}
    >
      <Toolbar />
      <List>
        {navItems.map(({ label, href, icon: Icon }) => (
          <ListItemButton key={href} component={Link} href={href} selected={pathname.startsWith(href)}>
            <ListItemIcon><Icon /></ListItemIcon>
            <ListItemText primary={label} />
          </ListItemButton>
        ))}
      </List>
    </Drawer>
  );
}
```

**`frontend/src/components/layout/AppShell.tsx`**

```tsx
"use client";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Sidebar, { DRAWER_WIDTH } from "./Sidebar";

export default function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <Box sx={{ display: "flex" }}>
      <AppBar position="fixed" sx={{ zIndex: (t) => t.zIndex.drawer + 1 }}>
        <Toolbar>
          <Typography variant="h6" noWrap>Research Intelligence Platform</Typography>
        </Toolbar>
      </AppBar>
      <Sidebar />
      <Box component="main" sx={{ flexGrow: 1, p: 3, minWidth: 0 }}>
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}
```

### 3.5 API client — one file per resource

**`frontend/src/lib/api/client.ts`**

```ts
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const isForm = init.body instanceof FormData;
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: isForm ? init.headers : { "Content-Type": "application/json", ...init.headers },
    cache: "no-store",
  });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json() as Promise<T>;
}
```

**`frontend/src/types/paper.ts`**

```ts
export interface Paper {
  id: string;
  source: string;
  source_id: string;
  title: string;
  abstract: string | null;
  year: number | null;
  status: string;
  created_at: string;
}

export interface PaperList {
  total: number;
  items: Paper[];
}
```

**`frontend/src/lib/api/papers.ts`**

```ts
import { apiFetch } from "./client";
import type { PaperList, Paper } from "@/types/paper";

export const listPapers = (limit = 20, offset = 0) =>
  apiFetch<PaperList>(`/papers?limit=${limit}&offset=${offset}`);

export const getPaper = (id: string) => apiFetch<Paper>(`/papers/${id}`);
```

### 3.6 First page

**`frontend/src/components/papers/PapersTable.tsx`**

```tsx
"use client";
import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import { listPapers } from "@/lib/api/papers";
import type { PaperList } from "@/types/paper";

export default function PapersTable() {
  const [data, setData] = useState<PaperList | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listPapers().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!data) return <CircularProgress />;

  return (
    <TableContainer component={Paper}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Title</TableCell>
            <TableCell>Year</TableCell>
            <TableCell>Source</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.items.map((p) => (
            <TableRow key={p.id} hover>
              <TableCell>{p.title}</TableCell>
              <TableCell>{p.year ?? "—"}</TableCell>
              <TableCell>{p.source}</TableCell>
              <TableCell><Chip size="small" label={p.status} /></TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
```

**`frontend/src/app/papers/page.tsx`**

```tsx
import Typography from "@mui/material/Typography";
import PapersTable from "@/components/papers/PapersTable";

export default function PapersPage() {
  return (
    <>
      <Typography variant="h5" gutterBottom>Papers</Typography>
      <PapersTable />
    </>
  );
}
```

**`frontend/src/app/page.tsx`** — redirect home to papers

```tsx
import { redirect } from "next/navigation";

export default function Home() {
  redirect("/papers");
}
```

### Verify

```bash
docker compose up -d --build
```
Open **http://localhost:3000** → Papers table (empty but no error). If you see a network/CORS error, check `CORS_ORIGINS` in `.env` and restart the backend.

### Learn
- Why do `AppShell`, `Sidebar` and `PapersTable` start with `"use client"`? (They use hooks / browser-only features. Server components cannot.)
- Why does `apiFetch` use `cache: "no-store"`? (Next.js caches `fetch` aggressively by default; research data changes.)

**Commit:** `phase 3: frontend skeleton`

---

# Phase 4 — Gathering data

This is the foundation of the whole project. Weak data = weak system. Do this carefully.

### 4.1 Decide your corpus first

Pick **one narrow research domain** (for example *"machine learning for cultural heritage"*, *"retrieval-augmented generation"*, or *"crop disease detection"*). Why narrow?

- Contradictions and gaps only make sense when papers discuss the *same* problem.
- Evaluation is feasible when you can recognise correct answers.
- 300–1,000 papers in one domain beats 10,000 random ones.

Write your domain and 10–15 seed queries in `LEARNING_LOG.md`.

### 4.2 Where papers come from

| Source | What you get | Access | Notes |
|--------|--------------|--------|-------|
| **arXiv API** | Metadata + free PDFs | Free, no key | **Max 1 request / 3 seconds.** Returns Atom XML. Start here. |
| **OpenAlex** | Metadata, citations, concepts, open-access links | Free, no key (add `mailto=` for the polite pool) | Great for citation graph + filtering by year/field. |
| **Semantic Scholar Graph API** | References, citations, TL;DR, influential citations | Free (key optional, higher limits) | Use to enrich the citation graph in Phase 9. |
| **PubMed Central (OA subset)** | Biomedical full text | Free | Use if your domain is health. |
| **Unpaywall** | Legal open-access PDF location for a DOI | Free, needs email | Fill missing PDFs. |
| **CORE** | Open-access aggregator | Free key | Large coverage. |
| **Manual upload** | Any PDF you own | Your UI | Always support this. |

### 4.3 Rules for responsible collection
1. **Respect rate limits** (sleep between requests; use `User-Agent` with your email).
2. **Only store open-access PDFs.** Keep the license in metadata if available.
3. **Be idempotent.** Re-running a collection must not duplicate rows (we added the unique constraint for this).
4. **Store metadata first, PDF second.** If a download fails you still have the record and can retry (`status` tracks this).
5. **Keep raw files.** Never overwrite a downloaded PDF; parsing can be redone, downloading can't always be.

### 4.4 Design: a collector interface

Every source implements the same two methods. Adding OpenAlex later means adding one file.

**`backend/requirements.txt`** — add:

```txt
feedparser>=6.0
```
(then `docker compose up -d --build backend`)

**`backend/app/ingestion/collectors/base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PaperMeta:
    source: str
    source_id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    doi: str | None = None
    pdf_url: str | None = None
    authors: list[str] = field(default_factory=list)


class BaseCollector(ABC):
    source: str

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[PaperMeta]:
        """Return paper metadata for a query."""

    @abstractmethod
    def download_pdf(self, meta: PaperMeta, dest_dir: str) -> str | None:
        """Download the PDF and return its local path (or None)."""
```

**`backend/app/ingestion/collectors/arxiv.py`**

```python
import os
import time

import feedparser
import httpx

from app.ingestion.collectors.base import BaseCollector, PaperMeta

ARXIV_API = "https://export.arxiv.org/api/query"
HEADERS = {"User-Agent": "research-intelligence-platform/0.1 (mailto:you@example.com)"}


class ArxivCollector(BaseCollector):
    source = "arxiv"

    def search(self, query: str, max_results: int = 25) -> list[PaperMeta]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        resp = httpx.get(ARXIV_API, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        time.sleep(3)  # arXiv politeness rule
        return [self._to_meta(entry) for entry in feed.entries]

    def download_pdf(self, meta: PaperMeta, dest_dir: str) -> str | None:
        if not meta.pdf_url:
            return None
        os.makedirs(dest_dir, exist_ok=True)
        safe_id = meta.source_id.replace("/", "_")
        path = os.path.join(dest_dir, f"arxiv_{safe_id}.pdf")
        if os.path.exists(path):          # never download twice
            return path
        with httpx.stream("GET", meta.pdf_url, headers=HEADERS, timeout=60, follow_redirects=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for block in r.iter_bytes():
                    f.write(block)
        time.sleep(3)
        return path

    @staticmethod
    def _to_meta(entry) -> PaperMeta:
        arxiv_id = entry.id.split("/abs/")[-1]
        pdf_url = next((l.href for l in entry.links if l.get("type") == "application/pdf"), None)
        return PaperMeta(
            source="arxiv",
            source_id=arxiv_id,
            title=" ".join(entry.title.split()),
            abstract=" ".join(entry.summary.split()),
            year=int(entry.published[:4]),
            doi=entry.get("arxiv_doi"),
            pdf_url=pdf_url,
            authors=[a.name for a in entry.get("authors", [])],
        )
```

### 4.5 Authors table (needed now, used by the graph later)

**`backend/app/models/author.py`**

```python
import uuid

from sqlalchemy import ForeignKey, String, Table, Column
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

paper_authors = Table(
    "paper_authors",
    Base.metadata,
    Column("paper_id", UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True),
    Column("author_id", UUID(as_uuid=True), ForeignKey("authors.id", ondelete="CASCADE"), primary_key=True),
)


class Author(Base):
    __tablename__ = "authors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), unique=True)
```

Add to `models/__init__.py`: `from app.models.author import Author, paper_authors  # noqa: F401`
Then run: `alembic revision --autogenerate -m "authors"` and `alembic upgrade head`.

### 4.6 Repository + service + endpoints

**`backend/app/repositories/paper_repository.py`** — add:

```python
from sqlalchemy.dialects.postgresql import insert

from app.models.author import Author, paper_authors


def upsert_paper(db: Session, meta) -> Paper:
    """Insert a paper or return the existing one (idempotent)."""
    stmt = (
        insert(Paper)
        .values(
            source=meta.source, source_id=meta.source_id, title=meta.title,
            abstract=meta.abstract, year=meta.year, doi=meta.doi, pdf_url=meta.pdf_url,
        )
        .on_conflict_do_nothing(constraint="uq_paper_source")
    )
    db.execute(stmt)
    db.commit()
    return db.scalar(select(Paper).where(Paper.source == meta.source, Paper.source_id == meta.source_id))


def attach_authors(db: Session, paper: Paper, names: list[str]) -> None:
    for name in names:
        author = db.scalar(select(Author).where(Author.name == name))
        if author is None:
            author = Author(name=name)
            db.add(author)
            db.flush()
        db.execute(
            insert(paper_authors)
            .values(paper_id=paper.id, author_id=author.id)
            .on_conflict_do_nothing()
        )
    db.commit()


def set_pdf_path(db: Session, paper: Paper, path: str) -> None:
    paper.pdf_path = path
    paper.status = "downloaded"
    db.commit()
```

**`backend/app/schemas/collect.py`**

```python
from pydantic import BaseModel, Field


class CollectRequest(BaseModel):
    query: str = Field(min_length=3, examples=["retrieval augmented generation"])
    max_results: int = Field(25, ge=1, le=100)
    download_pdfs: bool = True


class CollectResponse(BaseModel):
    found: int
    message: str
```

**`backend/app/services/collect_arxiv.py`**

```python
import os

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.ingestion.collectors.arxiv import ArxivCollector
from app.repositories import paper_repository as repo
from app.schemas.collect import CollectRequest, CollectResponse


def collect_metadata(db: Session, req: CollectRequest) -> tuple[CollectResponse, list]:
    """Step 1 (fast): fetch metadata and save it."""
    collector = ArxivCollector()
    metas = collector.search(req.query, req.max_results)
    for meta in metas:
        paper = repo.upsert_paper(db, meta)
        repo.attach_authors(db, paper, meta.authors)
    return CollectResponse(found=len(metas), message="Metadata saved; PDFs downloading in background"), metas


def download_pdfs(metas: list) -> None:
    """Step 2 (slow): runs in a background task with its own DB session."""
    collector = ArxivCollector()
    dest = os.path.join(get_settings().data_dir, "pdfs")
    db = SessionLocal()
    try:
        for meta in metas:
            try:
                path = collector.download_pdf(meta, dest)
                if path:
                    paper = repo.upsert_paper(db, meta)
                    repo.set_pdf_path(db, paper, path)
            except Exception as exc:        # one bad PDF must not stop the batch
                print(f"[download] {meta.source_id} failed: {exc}")
    finally:
        db.close()
```

**`backend/app/api/v1/endpoints/collect_arxiv.py`**

```python
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.collect import CollectRequest, CollectResponse
from app.services import collect_arxiv as svc

router = APIRouter()


@router.post("/collect/arxiv", response_model=CollectResponse, tags=["collect"])
def collect_arxiv(req: CollectRequest, tasks: BackgroundTasks, db: Session = Depends(get_db)):
    response, metas = svc.collect_metadata(db, req)
    if req.download_pdfs:
        tasks.add_task(svc.download_pdfs, metas)
    return response
```

Register `collect_arxiv` in `router.py`'s tuple.

**Manual upload — `backend/app/api/v1/endpoints/papers_upload.py`**

```python
import hashlib
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.ingestion.collectors.base import PaperMeta
from app.repositories import paper_repository as repo
from app.schemas.paper import PaperRead

router = APIRouter()


@router.post("/papers/upload", response_model=PaperRead, tags=["papers"])
async def upload_paper(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are accepted")
    content = await file.read()
    digest = hashlib.sha256(content).hexdigest()[:16]       # dedupe by content
    folder = os.path.join(get_settings().data_dir, "pdfs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"upload_{digest}.pdf")
    with open(path, "wb") as f:
        f.write(content)
    meta = PaperMeta(source="upload", source_id=digest, title=file.filename or digest)
    paper = repo.upsert_paper(db, meta)
    repo.set_pdf_path(db, paper, path)
    return PaperRead.model_validate(paper)
```

### Verify

```bash
curl -X POST http://localhost:8000/api/v1/collect/arxiv \
  -H "Content-Type: application/json" \
  -d '{"query":"retrieval augmented generation","max_results":10}'

# wait ~1 minute, then
curl "http://localhost:8000/api/v1/papers?limit=5"
ls data/pdfs
```
Run the same collect call again: `total` must **not** grow (idempotency).

### Frontend task (small)
Add `frontend/src/lib/api/collect.ts` (`POST /collect/arxiv`) and a `CollectDialog` component with a text field and a button; open it from the Papers page. This is your first mutation from the UI.

**Commit:** `phase 4: data collection`

---

# Phase 5 — Parsing PDFs and chunking

### What and why
PDFs are *layout*, not *structure*. To cite precisely and to compare findings you need: title, abstract, sections, tables, figure captions, and the reference list. You'll use **three tools, each for what it's best at**:

| Tool | Best at | Output |
|------|---------|--------|
| **GROBID** (ML service) | Sections, header, **references** | TEI XML |
| **PyMuPDF** (`fitz`) | Fast text per **page** (for page-number citations) and fallback | Plain text |
| **pdfplumber** | **Tables** | Rows → Markdown |

### 5.1 Add GROBID to Docker

```yaml
  grobid:
    image: lfoppiano/grobid:0.8.1       # check the tag on Docker Hub if it fails
    container_name: rip-grobid
    ports:
      - "8070:8070"
    ulimits:
      core: 0
    init: true
```

```bash
docker compose up -d grobid
curl http://localhost:8070/api/isalive      # true
```

Add to `requirements.txt`: `pymupdf>=1.24`, `pdfplumber>=0.11`, `lxml>=5.2`. Rebuild the backend.

Add to `.env`/`.env.example`: `GROBID_URL=http://grobid:8070` and add `grobid_url: str = "http://grobid:8070"` to `Settings`.

### 5.2 Data shapes (shared by parsers)

**`backend/app/ingestion/parsers/types.py`**

```python
from dataclasses import dataclass, field


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class Reference:
    title: str
    year: int | None = None
    doi: str | None = None
    authors: list[str] = field(default_factory=list)


@dataclass
class ParsedPaper:
    title: str | None
    abstract: str | None
    sections: list[Section]
    references: list[Reference]
```

### 5.3 GROBID client + TEI parser

**`backend/app/ingestion/parsers/grobid_client.py`**

```python
import httpx

from app.core.config import get_settings


def pdf_to_tei(pdf_path: str) -> str:
    """Send a PDF to GROBID and return TEI XML text."""
    url = f"{get_settings().grobid_url}/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        resp = httpx.post(url, files={"input": f}, data={"consolidateHeader": "0"}, timeout=180)
    resp.raise_for_status()
    return resp.text
```

**`backend/app/ingestion/parsers/tei_parser.py`**

```python
from lxml import etree

from app.ingestion.parsers.types import ParsedPaper, Reference, Section

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _text(node) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def parse_tei(tei_xml: str) -> ParsedPaper:
    root = etree.fromstring(tei_xml.encode("utf-8"))

    title = _text(root.find(".//tei:titleStmt/tei:title", NS)) or None
    abstract = _text(root.find(".//tei:profileDesc/tei:abstract", NS)) or None

    sections = []
    for div in root.findall(".//tei:text/tei:body/tei:div", NS):
        heading = _text(div.find("tei:head", NS)) or "Untitled"
        body = " ".join(_text(p) for p in div.findall("tei:p", NS))
        if body:
            sections.append(Section(heading=heading, text=body))

    references = []
    for bib in root.findall(".//tei:listBibl/tei:biblStruct", NS):
        ref_title = _text(bib.find(".//tei:title", NS))
        if not ref_title:
            continue
        date = bib.find(".//tei:date[@when]", NS)
        year = int(date.get("when")[:4]) if date is not None and date.get("when")[:4].isdigit() else None
        doi = _text(bib.find(".//tei:idno[@type='DOI']", NS)) or None
        authors = [_text(p) for p in bib.findall(".//tei:author/tei:persName", NS)]
        references.append(Reference(title=ref_title, year=year, doi=doi, authors=authors))

    return ParsedPaper(title=title, abstract=abstract, sections=sections, references=references)
```

### 5.4 PyMuPDF (pages) and pdfplumber (tables)

**`backend/app/ingestion/parsers/pymupdf_parser.py`**

```python
import fitz  # PyMuPDF


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text)] starting at page 1."""
    with fitz.open(pdf_path) as doc:
        return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
```

**`backend/app/ingestion/parsers/table_parser.py`**

```python
import pdfplumber


def extract_tables_markdown(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, markdown_table)] for every table found."""
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in table if row]
                if len(rows) < 2:
                    continue
                header, *body = rows
                md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                md += ["| " + " | ".join(r) + " |" for r in body]
                results.append((page_no, "\n".join(md)))
    return results
```

### 5.5 Figure captions (cheap, effective)

Extract images later if you need them. For retrieval, **captions are what matter**.

**`backend/app/ingestion/parsers/caption_parser.py`**

```python
import re

CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s*\d+[.:]\s*.+", re.IGNORECASE | re.MULTILINE)


def extract_captions(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    out = []
    for page_no, text in pages:
        for match in CAPTION_RE.finditer(text):
            out.append((page_no, " ".join(match.group(0).split())))
    return out
```

### 5.6 Chunking

**Why chunk at all?** An embedding of a whole paper is a blurry average. Small, focused chunks retrieve precisely and fit in the LLM prompt.

**Strategy:** chunk *within* sections (never across), ~300 words with 50-word overlap. Overlap prevents a sentence about "the proposed method" from being cut away from its explanation.

**`backend/app/ingestion/chunking/section_chunker.py`**

```python
from dataclasses import dataclass

from app.ingestion.parsers.types import ParsedPaper


@dataclass
class ChunkDraft:
    chunk_type: str          # abstract | text | table | caption
    section: str | None
    page: int | None
    text: str


def split_words(text: str, max_words: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - overlap
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), step)]


def chunk_paper(
    parsed: ParsedPaper,
    tables: list[tuple[int, str]],
    captions: list[tuple[int, str]],
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    if parsed.abstract:
        drafts.append(ChunkDraft("abstract", "Abstract", None, parsed.abstract))
    for section in parsed.sections:
        for piece in split_words(section.text):
            drafts.append(ChunkDraft("text", section.heading, None, piece))
    for page, md in tables:
        drafts.append(ChunkDraft("table", None, page, md))
    for page, cap in captions:
        drafts.append(ChunkDraft("caption", None, page, cap))
    return drafts
```

> Words ≈ tokens/0.75. Using words keeps the code simple; later you can switch to the tokenizer of your embedding model for exact limits.

### 5.7 Models for chunks and references

**`backend/app/models/chunk.py`** (the `embedding` column arrives in Phase 6)

```python
import uuid

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_type: Mapped[str] = mapped_column(String(20))
    section: Mapped[str | None] = mapped_column(String(300), nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)
```

**`backend/app/models/reference.py`**

```python
import uuid

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Reference(Base):
    """A bibliography entry found inside a paper (paper -> cites -> reference)."""

    __tablename__ = "references"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(200), nullable=True)
```

Register both in `models/__init__.py`, then migrate (`autogenerate` + `upgrade head`).

### Verify (in a Python shell inside the container)

```bash
docker compose exec backend python
>>> from app.ingestion.parsers.grobid_client import pdf_to_tei
>>> from app.ingestion.parsers.tei_parser import parse_tei
>>> p = parse_tei(pdf_to_tei("/data/pdfs/<one file>.pdf"))
>>> p.title, len(p.sections), len(p.references)
```
Open the PDF yourself and check: do the section names match? Are references plausible? Parsers are never perfect — noticing the errors is the lesson.

**Commit:** `phase 5: parsing and chunking`

---
# Phase 6 — Embeddings and vector search

### What and why
An **embedding model** turns each chunk into a vector. Chunks with similar meaning end up close together. Storing vectors in Postgres (via **pgvector**) means one database handles both your relational data and similarity search.

### Technologies
| Tool | Purpose |
|------|---------|
| **sentence-transformers** | Loads Hugging Face embedding models in two lines |
| **PyTorch** | The numeric engine under the model (CPU build is fine) |
| **BAAI/bge-small-en-v1.5** | Small, fast, strong English embedding model → **384 dimensions** |
| **pgvector + HNSW index** | Approximate nearest-neighbour search that stays fast with many chunks |

### 6.1 Dependencies

`requirements.txt` — add (CPU-only PyTorch keeps the image smaller):

```txt
--extra-index-url https://download.pytorch.org/whl/cpu
torch
sentence-transformers>=3.0
numpy
```
Rebuild: `docker compose up -d --build backend` (first build is large — this is normal).

Persist downloaded models between rebuilds by adding a volume to the backend service:

```yaml
    volumes:
      - ./backend:/code
      - ./data:/data
      - hf_cache:/root/.cache/huggingface
```
and under the top-level `volumes:` add `hf_cache:`.

### 6.2 Embedder (loads the model once)

**`backend/app/ingestion/embedding/embedder.py`**

```python
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
# bge models work best when *queries* get this prefix (documents do not)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    vectors = _model().encode(texts, batch_size=32, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return _model().encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
```

> `normalize_embeddings=True` makes every vector length 1, so cosine similarity equals dot product and pgvector's cosine operator behaves predictably.

### 6.3 Add the vector column

Edit **`models/chunk.py`**:

```python
from pgvector.sqlalchemy import Vector
from app.ingestion.embedding.embedder import EMBEDDING_DIM
# ...inside class Chunk:
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
```

Generate the migration, then **edit it** to add the HNSW index:

```bash
docker compose exec backend alembic revision --autogenerate -m "chunk embeddings"
```

In the generated file's `upgrade()` add:

```python
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
```
and in `downgrade()`: `op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")`. Then `alembic upgrade head`.

### 6.4 Repositories

**`backend/app/repositories/chunk_repository.py`**

```python
import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models.chunk import Chunk


def replace_chunks(db: Session, paper_id: uuid.UUID, drafts: list, vectors: list[list[float]]) -> list[Chunk]:
    """Delete old chunks for the paper and insert fresh ones (re-ingest safe)."""
    db.execute(delete(Chunk).where(Chunk.paper_id == paper_id))
    rows = [
        Chunk(
            paper_id=paper_id, chunk_index=i, chunk_type=d.chunk_type,
            section=d.section, page=d.page, text=d.text, embedding=vec,
        )
        for i, (d, vec) in enumerate(zip(drafts, vectors))
    ]
    db.add_all(rows)
    db.commit()
    return rows
```

**`backend/app/repositories/vector_repository.py`**

```python
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.paper import Paper


def search_vectors(db: Session, query_vector: list[float], k: int = 20) -> list[dict]:
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(Chunk, Paper.title, distance)
        .join(Paper, Paper.id == Chunk.paper_id)
        .order_by(distance)
        .limit(k)
    )
    return [
        {
            "chunk_id": str(c.id), "paper_id": str(c.paper_id), "paper_title": title,
            "section": c.section, "page": c.page, "text": c.text,
            "score": 1 - float(dist),              # cosine similarity
        }
        for c, title, dist in db.execute(stmt)
    ]
```

### 6.5 Ingestion service (the pipeline)

**`backend/app/services/ingest_paper.py`**

```python
import uuid

from sqlalchemy.orm import Session

from app.ingestion.chunking.section_chunker import chunk_paper
from app.ingestion.embedding.embedder import embed_documents
from app.ingestion.parsers.caption_parser import extract_captions
from app.ingestion.parsers.grobid_client import pdf_to_tei
from app.ingestion.parsers.pymupdf_parser import extract_pages
from app.ingestion.parsers.table_parser import extract_tables_markdown
from app.ingestion.parsers.tei_parser import parse_tei
from app.repositories import chunk_repository, paper_repository


def ingest_paper(db: Session, paper_id: uuid.UUID) -> int:
    paper = paper_repository.get_paper(db, paper_id)
    if paper is None or not paper.pdf_path:
        raise ValueError("Paper has no PDF to ingest")
    try:
        parsed = parse_tei(pdf_to_tei(paper.pdf_path))
        pages = extract_pages(paper.pdf_path)
        drafts = chunk_paper(parsed, extract_tables_markdown(paper.pdf_path), extract_captions(pages))
        if not drafts:                                   # GROBID found no sections: fall back
            drafts = _fallback_from_pages(pages)
        vectors = embed_documents([d.text for d in drafts])
        rows = chunk_repository.replace_chunks(db, paper.id, drafts, vectors)
        paper_repository.save_references(db, paper.id, parsed.references)
        paper.status = "parsed"
        db.commit()
        return len(rows)
    except Exception:
        paper.status = "failed"
        db.commit()
        raise


def _fallback_from_pages(pages):
    from app.ingestion.chunking.section_chunker import ChunkDraft, split_words
    return [ChunkDraft("text", None, no, piece) for no, text in pages for piece in split_words(text)]
```

Add `save_references` to `paper_repository.py`:

```python
from app.models.reference import Reference as ReferenceRow


def save_references(db: Session, paper_id, refs) -> None:
    db.query(ReferenceRow).filter(ReferenceRow.paper_id == paper_id).delete()
    db.add_all([ReferenceRow(paper_id=paper_id, title=r.title, year=r.year, doi=r.doi) for r in refs])
    db.commit()
```

**Endpoints**

`api/v1/endpoints/ingest_paper.py`

```python
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.db.session import SessionLocal, get_db
from app.services.ingest_paper import ingest_paper

router = APIRouter()


def _run(paper_id: uuid.UUID) -> None:
    db = SessionLocal()
    try:
        ingest_paper(db, paper_id)
    finally:
        db.close()


@router.post("/ingest/{paper_id}", status_code=202, tags=["ingest"])
def ingest(paper_id: uuid.UUID, tasks: BackgroundTasks) -> dict:
    tasks.add_task(_run, paper_id)
    return {"paper_id": str(paper_id), "status": "queued"}
```

`api/v1/endpoints/ingest_all.py` — ingest every downloaded paper

```python
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.v1.endpoints.ingest_paper import _run
from app.db.session import get_db
from app.models.paper import Paper

router = APIRouter()


@router.post("/ingest", status_code=202, tags=["ingest"])
def ingest_all(tasks: BackgroundTasks, db: Session = Depends(get_db)) -> dict:
    ids = list(db.scalars(select(Paper.id).where(Paper.status == "downloaded")))
    for pid in ids:
        tasks.add_task(_run, pid)
    return {"queued": len(ids)}
```

`api/v1/endpoints/search_vector.py` — plus its schemas

`schemas/search.py`

```python
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    k: int = Field(10, ge=1, le=50)


class SearchHit(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    section: str | None
    page: int | None
    text: str
    score: float


class SearchResponse(BaseModel):
    method: str
    hits: list[SearchHit]
```

`services/vector_search.py`

```python
from sqlalchemy.orm import Session

from app.ingestion.embedding.embedder import embed_query
from app.repositories import vector_repository
from app.schemas.search import SearchHit, SearchRequest, SearchResponse


def vector_search(db: Session, req: SearchRequest) -> SearchResponse:
    rows = vector_repository.search_vectors(db, embed_query(req.query), req.k)
    return SearchResponse(method="vector", hits=[SearchHit(**r) for r in rows])
```

`api/v1/endpoints/search_vector.py`

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse
from app.services.vector_search import vector_search

router = APIRouter()


@router.post("/search/vector", response_model=SearchResponse, tags=["search"])
def search_vector(req: SearchRequest, db: Session = Depends(get_db)):
    return vector_search(db, req)
```

Register `ingest_paper`, `ingest_all`, `search_vector` in `router.py`.

### Verify

```bash
curl -X POST localhost:8000/api/v1/ingest          # ingest all downloaded PDFs
docker compose logs -f backend                     # watch progress
curl -X POST localhost:8000/api/v1/search/vector -H "Content-Type: application/json" \
     -d '{"query":"how does retrieval improve factuality?","k":5}'
```
Read the top 5 results. Are they relevant? Try a query with an acronym (e.g. an exact method name). Vector search will often struggle there — that's your motivation for Phase 7.

**Commit:** `phase 6: embeddings and vector search`

---

# Phase 7 — BM25, hybrid retrieval, reranking

### What and why
- **Vector search** understands meaning but can miss exact names, numbers, acronyms.
- **BM25** matches exact words but misses paraphrases.
- **Hybrid** = both, merged. **Reranking** then polishes the top candidates with a smarter model.

Pipeline: `question → [BM25 top 50] + [vector top 50] → RRF merge → top 30 → cross-encoder rerank → top 8`

### Technologies
| Tool | Purpose |
|------|---------|
| **OpenSearch 2.x** | Open-source Elasticsearch fork; BM25 is its default ranking |
| **opensearch-py** | Python client |
| **RRF** | Rank-based merge (our own 10-line function) |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | Reranker from Hugging Face |

### 7.1 Add OpenSearch to Docker

```yaml
  opensearch:
    image: opensearchproject/opensearch:2.15.0
    container_name: rip-opensearch
    environment:
      - discovery.type=single-node
      - DISABLE_SECURITY_PLUGIN=true        # local dev only!
      - OPENSEARCH_INITIAL_ADMIN_PASSWORD=${OPENSEARCH_ADMIN_PASSWORD}
      - OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m
    ulimits:
      memlock: { soft: -1, hard: -1 }
    ports:
      - "9200:9200"
    volumes:
      - osdata:/usr/share/opensearch/data
```
Add `osdata:` under `volumes:`. Add to `.env`: `OPENSEARCH_ADMIN_PASSWORD=Str0ng!Passw0rd_ChangeMe`, `OPENSEARCH_URL=http://opensearch:9200`, and the matching `opensearch_url` setting in `config.py`.

```bash
docker compose up -d opensearch
curl localhost:9200          # JSON with version info
```
(Linux only, if it crashes: `sudo sysctl -w vm.max_map_count=262144`.)

`requirements.txt`: add `opensearch-py>=2.6`.

### 7.2 Index management

**`backend/app/repositories/bm25_repository.py`**

```python
from functools import lru_cache

from opensearchpy import OpenSearch, helpers

from app.core.config import get_settings

INDEX = "chunks"

MAPPING = {
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "paper_id": {"type": "keyword"},
            "paper_title": {"type": "text", "analyzer": "english"},
            "section": {"type": "keyword"},
            "page": {"type": "integer"},
            "text": {"type": "text", "analyzer": "english"},   # BM25 is the default similarity
        }
    },
}


@lru_cache
def client() -> OpenSearch:
    return OpenSearch(hosts=[get_settings().opensearch_url])


def ensure_index() -> None:
    if not client().indices.exists(INDEX):
        client().indices.create(INDEX, body=MAPPING)


def index_chunks(docs: list[dict]) -> None:
    ensure_index()
    actions = [{"_index": INDEX, "_id": d["chunk_id"], "_source": d} for d in docs]
    helpers.bulk(client(), actions, refresh=True)


def delete_paper(paper_id: str) -> None:
    client().delete_by_query(INDEX, body={"query": {"term": {"paper_id": paper_id}}}, refresh=True)


def search_bm25(query: str, k: int = 20) -> list[dict]:
    body = {
        "size": k,
        "query": {
            "multi_match": {"query": query, "fields": ["text", "paper_title^0.5"]}
        },
    }
    res = client().search(index=INDEX, body=body)
    return [
        {**h["_source"], "score": h["_score"]} for h in res["hits"]["hits"]
    ]
```

Hook into ingestion — in `ingest_paper.py`, after `replace_chunks`:

```python
from app.repositories import bm25_repository

bm25_repository.delete_paper(str(paper.id))
bm25_repository.index_chunks([
    {"chunk_id": str(r.id), "paper_id": str(paper.id), "paper_title": paper.title,
     "section": r.section, "page": r.page, "text": r.text}
    for r in rows
])
paper.status = "indexed"          # instead of "parsed"
```

Re-ingest (`POST /ingest` only picks `downloaded`; temporarily reset status or call `POST /ingest/{id}` per paper) so existing chunks reach OpenSearch.

### 7.3 BM25 endpoint

`services/bm25_search.py`

```python
from app.repositories import bm25_repository
from app.schemas.search import SearchHit, SearchRequest, SearchResponse


def bm25_search(req: SearchRequest) -> SearchResponse:
    return SearchResponse(method="bm25", hits=[SearchHit(**h) for h in bm25_repository.search_bm25(req.query, req.k)])
```

`api/v1/endpoints/search_bm25.py` — same shape as `search_vector.py`, calling `bm25_search(req)`.

### 7.4 Reciprocal Rank Fusion

**`backend/app/retrieval/rrf.py`**

```python
def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge ranked lists. Each hit needs a 'chunk_id'. Score = sum of 1/(k + rank)."""
    fused: dict[str, dict] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(hit["chunk_id"], {**hit, "score": 0.0})
            entry["score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda h: h["score"], reverse=True)
```

> **Why ranks and not scores?** BM25 scores (e.g. 14.2) and cosine similarities (e.g. 0.78) live on different scales. Ranks are comparable.

### 7.5 Reranker

**`backend/app/retrieval/reranker.py`**

```python
from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache
def _model() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, hits: list[dict], top_n: int = 8) -> list[dict]:
    if not hits:
        return []
    scores = _model().predict([(query, h["text"]) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return [{**h, "score": float(s)} for h, s in ranked[:top_n]]
```

### 7.6 Hybrid service + endpoint

**`backend/app/services/hybrid_search.py`**

```python
from sqlalchemy.orm import Session

from app.ingestion.embedding.embedder import embed_query
from app.repositories import bm25_repository, vector_repository
from app.retrieval.reranker import rerank
from app.retrieval.rrf import reciprocal_rank_fusion


def hybrid_hits(db: Session, query: str, k: int = 8, candidates: int = 50, use_rerank: bool = True) -> list[dict]:
    """Reusable by /search/hybrid, /ask and the agent."""
    bm25 = bm25_repository.search_bm25(query, candidates)
    vec = vector_repository.search_vectors(db, embed_query(query), candidates)
    merged = reciprocal_rank_fusion([bm25, vec])[:30]
    return rerank(query, merged, k) if use_rerank else merged[:k]
```

`schemas/search.py` — add `use_rerank: bool = True` to `SearchRequest`.

**`api/v1/endpoints/search_hybrid.py`**

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.search import SearchHit, SearchRequest, SearchResponse
from app.services.hybrid_search import hybrid_hits

router = APIRouter()


@router.post("/search/hybrid", response_model=SearchResponse, tags=["search"])
def search_hybrid(req: SearchRequest, db: Session = Depends(get_db)):
    hits = hybrid_hits(db, req.query, req.k, use_rerank=req.use_rerank)
    return SearchResponse(method="hybrid+rerank" if req.use_rerank else "hybrid", hits=[SearchHit(**h) for h in hits])
```

### Verify
Run the **same 5 questions** against `/search/bm25`, `/search/vector`, `/search/hybrid` (with and without rerank). Write which method won each time in `LEARNING_LOG.md`. In Phase 11 you will measure this properly instead of eyeballing it.

**Commit:** `phase 7: hybrid retrieval`

---

# Phase 8 — Answers with citations and citation checking

### What and why
Retrieval gives evidence. Now the LLM must (1) answer **only** from that evidence, (2) mark each claim with a citation, and (3) you must **verify** each citation actually supports its claim. This is what separates a production RAG from "chat with PDFs".

### Technologies
| Tool | Purpose |
|------|---------|
| **Anthropic Python SDK** (or **Ollama** for local models) | LLM generation |
| **Prompt templates** | Keep prompts out of code, versionable |
| **Regex citation parser** | Extract `[1]`, `[2]` markers |
| **LLM-as-judge verifier** | Checks claim ↔ cited passage |

### 8.1 Configuration

`requirements.txt`: add `anthropic>=0.40`.

`.env`:

```env
LLM_PROVIDER=anthropic          # anthropic | ollama
ANTHROPIC_API_KEY=your_key_here
LLM_MODEL=claude-sonnet-5-5     # check docs.claude.com for current model names
OLLAMA_URL=http://host.docker.internal:11434
```
Add matching fields (`llm_provider`, `anthropic_api_key`, `llm_model`, `ollama_url`) to `Settings`.

### 8.2 LLM abstraction (swap providers without touching services)

**`backend/app/llm/base.py`**

```python
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, system: str, user: str, max_tokens: int = 1000) -> str: ...
```

**`backend/app/llm/anthropic_client.py`**

```python
import anthropic

from app.core.config import get_settings


class AnthropicClient:
    def __init__(self) -> None:
        s = get_settings()
        self._client = anthropic.Anthropic(api_key=s.anthropic_api_key)
        self._model = s.llm_model

    def generate(self, system: str, user: str, max_tokens: int = 1000) -> str:
        msg = self._client.messages.create(
            model=self._model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
```

**`backend/app/llm/ollama_client.py`**

```python
import httpx

from app.core.config import get_settings


class OllamaClient:
    def __init__(self) -> None:
        s = get_settings()
        self._url, self._model = s.ollama_url, s.llm_model

    def generate(self, system: str, user: str, max_tokens: int = 1000) -> str:
        resp = httpx.post(
            f"{self._url}/api/chat",
            json={"model": self._model, "stream": False,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                  "options": {"num_predict": max_tokens}},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
```

**`backend/app/llm/factory.py`**

```python
from functools import lru_cache

from app.core.config import get_settings
from app.llm.base import LLMClient


@lru_cache
def get_llm() -> LLMClient:
    if get_settings().llm_provider == "ollama":
        from app.llm.ollama_client import OllamaClient
        return OllamaClient()
    from app.llm.anthropic_client import AnthropicClient
    return AnthropicClient()
```

### 8.3 Prompts as files

**`backend/app/llm/prompts/answer_system.txt`**

```text
You are a careful research assistant. Answer the question using ONLY the numbered sources provided.
Rules:
1. After every sentence that states a fact from a source, add its citation like [1] or [2][3].
2. If the sources do not contain enough information, say exactly: "The provided papers do not contain enough evidence."
3. Never use outside knowledge. Never invent citations.
4. Be concise and precise. Prefer numbers and method names exactly as written in the sources.
```

**`backend/app/llm/prompts/loader.py`**

```python
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@lru_cache
def load_prompt(name: str) -> str:
    return (_DIR / f"{name}.txt").read_text(encoding="utf-8")
```

### 8.4 Ask service

**`backend/app/schemas/ask.py`**

```python
from pydantic import BaseModel, Field

from app.schemas.search import SearchHit


class AskRequest(BaseModel):
    question: str = Field(min_length=5)
    k: int = Field(8, ge=1, le=20)
    verify: bool = True


class CitationCheck(BaseModel):
    sentence: str
    sources: list[int]
    supported: bool
    reason: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchHit]               # position + 1 == citation number
    citation_checks: list[CitationCheck]
    citation_accuracy: float | None        # supported / total checked
```

**`backend/app/verification/citation_parser.py`**

```python
import re

CITATION = re.compile(r"\[(\d+)\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_cited_sentences(answer: str) -> list[tuple[str, list[int]]]:
    """Return [(sentence, [citation numbers])] for sentences that contain citations."""
    out = []
    for sentence in SENTENCE_SPLIT.split(answer):
        nums = [int(n) for n in CITATION.findall(sentence)]
        if nums:
            out.append((sentence.strip(), nums))
    return out
```

**`backend/app/llm/prompts/verify_system.txt`**

```text
You verify citations. You are given a CLAIM and the SOURCE TEXT it cites.
Decide whether the source text directly supports the claim.
Reply with JSON only: {"supported": true|false, "reason": "<one short sentence>"}
```

**`backend/app/verification/citation_verifier.py`**

```python
import json

from app.llm.base import LLMClient
from app.llm.prompts.loader import load_prompt
from app.schemas.ask import CitationCheck
from app.verification.citation_parser import split_cited_sentences


def verify_citations(llm: LLMClient, answer: str, sources: list[dict]) -> list[CitationCheck]:
    checks = []
    for sentence, nums in split_cited_sentences(answer):
        valid = [n for n in nums if 1 <= n <= len(sources)]
        if not valid:                                   # citation to a source that doesn't exist
            checks.append(CitationCheck(sentence=sentence, sources=nums, supported=False, reason="Invalid citation number"))
            continue
        source_text = "\n\n".join(sources[n - 1]["text"] for n in valid)
        raw = llm.generate(load_prompt("verify_system"), f"CLAIM: {sentence}\n\nSOURCE TEXT:\n{source_text}", max_tokens=150)
        try:
            data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            checks.append(CitationCheck(sentence=sentence, sources=nums, supported=bool(data["supported"]), reason=data.get("reason", "")))
        except (ValueError, KeyError):
            checks.append(CitationCheck(sentence=sentence, sources=nums, supported=False, reason="Verifier output unparsable"))
    return checks
```

**`backend/app/services/answer_question.py`**

```python
from sqlalchemy.orm import Session

from app.llm.factory import get_llm
from app.llm.prompts.loader import load_prompt
from app.schemas.ask import AskRequest, AskResponse
from app.schemas.search import SearchHit
from app.services.hybrid_search import hybrid_hits
from app.verification.citation_verifier import verify_citations


def format_sources(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({h['paper_title']}, {h.get('section') or 'n/a'})\n{h['text']}" for i, h in enumerate(hits, 1))


def answer_question(db: Session, req: AskRequest) -> AskResponse:
    hits = hybrid_hits(db, req.question, req.k)
    llm = get_llm()
    prompt = f"SOURCES:\n{format_sources(hits)}\n\nQUESTION: {req.question}"
    answer = llm.generate(load_prompt("answer_system"), prompt, max_tokens=900)

    checks = verify_citations(llm, answer, hits) if req.verify else []
    accuracy = (sum(c.supported for c in checks) / len(checks)) if checks else None
    return AskResponse(answer=answer, sources=[SearchHit(**h) for h in hits], citation_checks=checks, citation_accuracy=accuracy)
```

**`api/v1/endpoints/ask.py`**

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.ask import AskRequest, AskResponse
from app.services.answer_question import answer_question

router = APIRouter()


@router.post("/ask", response_model=AskResponse, tags=["ask"])
def ask(req: AskRequest, db: Session = Depends(get_db)):
    return answer_question(db, req)
```

### Verify

```bash
curl -X POST localhost:8000/api/v1/ask -H "Content-Type: application/json" \
  -d '{"question":"What are the main limitations of retrieval-augmented generation?"}'
```
Check by hand: open one cited source — does it really say that? Then ask something your corpus **cannot** answer. The correct response is the "not enough evidence" sentence.

### Try next (compare + gaps endpoints)
Both are thin services over the same building blocks:

- `POST /compare` — body `{"question": "...", "paper_ids": [..]}`. For each paper, call `hybrid_hits` filtered to that paper (add an optional `paper_ids` filter to the two repositories), then ask the LLM for a table: *method, dataset, metric, result, limitation*, each cell cited.
- `POST /gaps` — implemented in Phase 10 using the knowledge graph.

**Commit:** `phase 8: grounded answers with citation checking`

---

# Phase 9 — Knowledge graph and graph retrieval

### What and why
Vector/BM25 find *passages*. A graph answers *relationship* questions: "Which methods were evaluated on dataset X?", "Who works on both A and B?", "Which papers cite this one?" It also powers **gap detection** (method × dataset combinations nobody tried).

### Technologies
| Tool | Purpose |
|------|---------|
| **LLM structured extraction** | Pull methods, datasets, tasks, metrics, findings out of each paper as JSON |
| **PostgreSQL tables (`entities`, `edges`)** | Simple graph store — no new infrastructure |
| **Recursive CTE** | Multi-hop traversal in SQL |
| **Semantic Scholar API** (optional) | Enrich citation edges between papers |
| *(Optional upgrade)* **Neo4j + Cypher** | Dedicated graph DB once you outgrow SQL |

### 9.1 Graph model

Node types: `paper`, `author`, `method`, `dataset`, `task`, `metric`.
Edge types: `AUTHORED`, `CITES`, `USES_METHOD`, `EVALUATED_ON`, `ADDRESSES_TASK`, `REPORTS_METRIC`.

**`backend/app/models/entity.py`**

```python
import uuid

from sqlalchemy import Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("kind", "name", name="uq_entity"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(20), index=True)     # method | dataset | task | metric | paper | author
    name: Mapped[str] = mapped_column(String(300), index=True)    # normalised (lower-case)
    ref_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)  # papers.id / authors.id


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (UniqueConstraint("src_id", "dst_id", "relation", "paper_id", name="uq_edge"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    src_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"), index=True)
    dst_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"), index=True)
    relation: Mapped[str] = mapped_column(String(30), index=True)
    paper_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)   # quote/finding supporting this edge
    weight: Mapped[float] = mapped_column(Float, default=1.0)
```

Register in `models/__init__.py`, migrate.

### 9.2 Extraction (LLM → JSON)

**`backend/app/graph_extraction/schemas.py`**

```python
from pydantic import BaseModel, Field


class Extraction(BaseModel):
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
```

**`backend/app/llm/prompts/extract_system.txt`**

```text
Extract structured facts from the research paper text. Reply with JSON only, with these keys:
methods, datasets, tasks, metrics, key_findings, limitations — each a list of short strings.
Use the exact names used in the paper. Do not guess; leave a list empty if absent.
```

**`backend/app/graph_extraction/extractor.py`**

```python
import json

from app.graph_extraction.schemas import Extraction
from app.llm.base import LLMClient
from app.llm.prompts.loader import load_prompt


def extract_entities(llm: LLMClient, title: str, abstract: str | None, section_texts: list[str]) -> Extraction:
    body = f"TITLE: {title}\nABSTRACT: {abstract or ''}\n\n" + "\n\n".join(section_texts)[:12000]
    raw = llm.generate(load_prompt("extract_system"), body, max_tokens=700)
    try:
        return Extraction.model_validate(json.loads(raw[raw.index("{") : raw.rindex("}") + 1]))
    except (ValueError, KeyError):
        return Extraction()
```

### 9.3 Repository + service + endpoints

**`backend/app/repositories/graph_repository.py`**

```python
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.entity import Edge, Entity


def normalise(name: str) -> str:
    return " ".join(name.lower().split())


def get_or_create_entity(db: Session, kind: str, name: str, ref_id=None) -> Entity:
    name = normalise(name)
    db.execute(insert(Entity).values(kind=kind, name=name, ref_id=ref_id).on_conflict_do_nothing(constraint="uq_entity"))
    return db.query(Entity).filter_by(kind=kind, name=name).one()


def add_edge(db: Session, src: Entity, dst: Entity, relation: str, paper_id, evidence: str | None = None) -> None:
    db.execute(
        insert(Edge)
        .values(src_id=src.id, dst_id=dst.id, relation=relation, paper_id=paper_id, evidence=evidence)
        .on_conflict_do_nothing(constraint="uq_edge")
    )


NEIGHBORS_SQL = text("""
WITH RECURSIVE walk AS (
    SELECT e.src_id, e.dst_id, e.relation, e.paper_id, 1 AS depth
    FROM edges e WHERE e.src_id = :node OR e.dst_id = :node
  UNION
    SELECT e.src_id, e.dst_id, e.relation, e.paper_id, w.depth + 1
    FROM edges e JOIN walk w ON e.src_id IN (w.src_id, w.dst_id) OR e.dst_id IN (w.src_id, w.dst_id)
    WHERE w.depth < :max_depth
)
SELECT DISTINCT src_id, dst_id, relation, paper_id FROM walk LIMIT :lim
""")


def neighbors(db: Session, node_id, max_depth: int = 2, limit: int = 200) -> list[dict]:
    rows = db.execute(NEIGHBORS_SQL, {"node": node_id, "max_depth": max_depth, "lim": limit})
    return [dict(r._mapping) for r in rows]


def papers_for_entities(db: Session, entity_ids: list) -> list:
    """Graph retrieval: papers connected to any of the given entities."""
    rows = db.execute(
        text("SELECT DISTINCT paper_id FROM edges WHERE (src_id = ANY(:ids) OR dst_id = ANY(:ids)) AND paper_id IS NOT NULL"),
        {"ids": entity_ids},
    )
    return [r[0] for r in rows]
```

> A `UNION` (not `UNION ALL`) in the recursive query prevents infinite loops on cyclic graphs, and `depth < :max_depth` bounds the walk.

**`backend/app/services/build_graph.py`** (per paper)

```python
import uuid

from sqlalchemy.orm import Session

from app.graph_extraction.extractor import extract_entities
from app.llm.factory import get_llm
from app.models.chunk import Chunk
from app.repositories import graph_repository as g
from app.repositories import paper_repository

KIND_TO_RELATION = {
    "methods": ("method", "USES_METHOD"),
    "datasets": ("dataset", "EVALUATED_ON"),
    "tasks": ("task", "ADDRESSES_TASK"),
    "metrics": ("metric", "REPORTS_METRIC"),
}


def build_graph_for_paper(db: Session, paper_id: uuid.UUID) -> dict:
    paper = paper_repository.get_paper(db, paper_id)
    texts = [c.text for c in db.query(Chunk).filter_by(paper_id=paper_id, chunk_type="text").limit(20)]
    extraction = extract_entities(get_llm(), paper.title, paper.abstract, texts)

    paper_node = g.get_or_create_entity(db, "paper", paper.title, ref_id=paper.id)
    counts = {}
    for field, (kind, relation) in KIND_TO_RELATION.items():
        names = getattr(extraction, field)
        for name in names:
            node = g.get_or_create_entity(db, kind, name)
            g.add_edge(db, paper_node, node, relation, paper.id)
        counts[field] = len(names)
    db.commit()
    return counts
```

**Endpoints (one file each):**

| File | Route | Purpose |
|------|-------|---------|
| `graph_build.py` | `POST /graph/build/{paper_id}` | Extract + store graph for one paper |
| `graph_build_all.py` | `POST /graph/build` | Background build for all indexed papers |
| `graph_search_entities.py` | `GET /graph/entities?q=bert&kind=method` | Autocomplete entities |
| `graph_neighbors.py` | `GET /graph/neighbors/{entity_id}?depth=2` | Subgraph for the UI |
| `graph_papers_by_entity.py` | `GET /graph/entities/{entity_id}/papers` | Papers linked to an entity |

Each follows the same 15-line pattern as `papers_get.py`.

### 9.4 Citation edges (paper → paper)
You already saved each paper's `references`. Match them to papers you hold, by DOI first and normalised title second, then create `CITES` edges:

```python
# services/link_citations.py (sketch)
# for each Reference row: find Paper where doi == ref.doi OR lower(title) == lower(ref.title)
# if found: add_edge(paper_node, cited_paper_node, "CITES", paper_id=ref.paper_id)
```
To enrich beyond your own corpus, call `https://api.graph.semanticscholar.org/graph/v1/paper/arXiv:{id}/references?fields=title,externalIds,year`.

### 9.5 Graph retrieval inside the pipeline
`hybrid_hits` gets an optional graph step: extract entities from the question (LLM or simple name matching against the `entities` table), call `papers_for_entities`, and **boost** hybrid hits whose `paper_id` is in that set (e.g. add 0.5 to the RRF score) or feed the graph papers' abstracts as extra candidates before reranking. Keep it a flag (`use_graph=True`) so Phase 11 can measure whether it helps.

### Verify
`POST /graph/build/{id}` on 3 papers, then in `psql`:

```sql
SELECT kind, name FROM entities WHERE kind='dataset' LIMIT 20;
SELECT relation, count(*) FROM edges GROUP BY relation;
```
Are extracted names sane? Do you see duplicates ("BERT" vs "BERT-base")? Normalisation/merging is a real engineering problem — note your findings.

**Commit:** `phase 9: knowledge graph`

---
# Phase 10 — Multi-step agent (LangGraph)

### What and why
`/ask` is a straight line: retrieve → answer. A real research question needs a **loop**: plan sub-questions, retrieve for each, judge whether evidence is enough, search again if not, look for contradictions, write, **verify citations**, rewrite if verification fails, then suggest research gaps.

**LangGraph** models this as a state machine: *nodes* (functions that read/update shared state) and *edges* (which node runs next, possibly conditional). Because every step is explicit, you can log, stream and test each one.

### Technologies
| Tool | Purpose |
|------|---------|
| **LangGraph** | Agent as a graph of nodes + conditional edges |
| **Server-Sent Events (SSE)** | One-way live stream of agent steps to the browser |
| **PostgreSQL (JSONB)** | Persist runs and events so the UI can replay them |

`requirements.txt`: add `langgraph>=0.2`.

### 10.1 The agent's shape

```
        ┌────────┐     ┌──────────┐     ┌───────┐  not enough evidence
 start →│  plan  │────▶│ retrieve │────▶│ grade │───────────┐
        └────────┘     └──────────┘     └───┬───┘           │ (max 2 loops)
                              ▲             │ enough        │
                              └─────────────┼───────────────┘
                                            ▼
                                    ┌───────────────┐
                                    │ contradictions│
                                    └───────┬───────┘
                                            ▼
                              ┌───────▶ ┌───────┐
                  rewrite once│         │ write │
                              │         └───┬───┘
                         ┌────┴───┐         ▼
                         │ verify │◀────────┘
                         └────┬───┘
                              ▼ ok
                          ┌───────┐
                          │ gaps  │──▶ END
                          └───────┘
```

### 10.2 State and dependencies

**`backend/app/agents/state.py`**

```python
from dataclasses import dataclass
from typing import TypedDict

from sqlalchemy.orm import Session

from app.llm.base import LLMClient


class AgentState(TypedDict, total=False):
    question: str
    plan: list[str]            # sub-queries to run next
    evidence: list[dict]       # deduplicated hits
    iterations: int
    sufficient: bool
    answer: str
    rewrites: int
    citation_checks: list[dict]
    citation_accuracy: float
    contradictions: list[dict]
    gaps: list[str]


@dataclass
class Deps:
    db: Session
    llm: LLMClient
```

**`backend/app/agents/json_utils.py`** (shared helper, one job)

```python
import json


def parse_json(raw: str, default):
    """Pull the first JSON object/array out of an LLM reply; return default if invalid."""
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        if open_ch in raw and close_ch in raw:
            try:
                return json.loads(raw[raw.index(open_ch) : raw.rindex(close_ch) + 1])
            except ValueError:
                continue
    return default
```

### 10.3 Nodes — one file each (`backend/app/agents/nodes/`)

**`plan.py`**

```python
from app.agents.json_utils import parse_json
from app.agents.state import AgentState, Deps

SYSTEM = (
    "Break the research question into 2-4 focused search queries for a paper database. "
    'Reply with JSON only: {"queries": ["...", "..."]}'
)


def plan_node(state: AgentState, deps: Deps) -> dict:
    data = parse_json(deps.llm.generate(SYSTEM, state["question"], 200), {"queries": [state["question"]]})
    return {"plan": data.get("queries") or [state["question"]], "iterations": 0, "evidence": [], "rewrites": 0}
```

**`retrieve.py`**

```python
from app.agents.state import AgentState, Deps
from app.services.hybrid_search import hybrid_hits


def retrieve_node(state: AgentState, deps: Deps) -> dict:
    seen = {h["chunk_id"] for h in state.get("evidence", [])}
    evidence = list(state.get("evidence", []))
    for query in state["plan"]:
        for hit in hybrid_hits(deps.db, query, k=5):
            if hit["chunk_id"] not in seen:
                seen.add(hit["chunk_id"])
                evidence.append(hit)
    return {"evidence": evidence, "iterations": state.get("iterations", 0) + 1}
```

**`grade.py`**

```python
from app.agents.json_utils import parse_json
from app.agents.state import AgentState, Deps

SYSTEM = (
    "Decide if the evidence is sufficient to answer the question. "
    'Reply JSON only: {"sufficient": true|false, "missing_query": "<search query for what is missing, or empty>"}'
)


def grade_node(state: AgentState, deps: Deps) -> dict:
    ev = "\n\n".join(f"- {h['text'][:400]}" for h in state["evidence"][:15])
    data = parse_json(deps.llm.generate(SYSTEM, f"QUESTION: {state['question']}\n\nEVIDENCE:\n{ev}", 150), {"sufficient": True})
    update = {"sufficient": bool(data.get("sufficient", True))}
    if not update["sufficient"] and data.get("missing_query"):
        update["plan"] = [data["missing_query"]]
    return update
```

**`contradictions.py`**

```python
from app.agents.json_utils import parse_json
from app.agents.state import AgentState, Deps

SYSTEM = (
    "Find claims in the numbered sources that CONTRADICT each other (different papers, opposite or incompatible findings). "
    'Reply JSON only: {"contradictions": [{"claim_a": "", "source_a": 1, "claim_b": "", "source_b": 2, "explanation": ""}]}. '
    'Return {"contradictions": []} if there are none. Do not invent.'
)


def contradictions_node(state: AgentState, deps: Deps) -> dict:
    sources = state["evidence"][:12]
    text = "\n\n".join(f"[{i}] ({h['paper_title']}) {h['text'][:600]}" for i, h in enumerate(sources, 1))
    data = parse_json(deps.llm.generate(SYSTEM, text, 700), {"contradictions": []})
    return {"contradictions": data.get("contradictions", [])}
```

> **Improvement later:** replace the single LLM call with an NLI model (`cross-encoder/nli-deberta-v3-base`) run on claim pairs extracted from the graph's `key_findings` of papers addressing the same task.

**`write.py`**

```python
from app.agents.state import AgentState, Deps
from app.llm.prompts.loader import load_prompt
from app.services.answer_question import format_sources


def write_node(state: AgentState, deps: Deps) -> dict:
    sources = state["evidence"][:10]
    note = ""
    if state.get("contradictions"):
        note = "\nNote: sources contain conflicting findings. State both sides and cite each.\n"
    if state.get("citation_checks"):
        bad = [c["sentence"] for c in state["citation_checks"] if not c["supported"]]
        if bad:
            note += "\nThese sentences were NOT supported by their citations; fix or remove them:\n- " + "\n- ".join(bad)
    prompt = f"SOURCES:\n{format_sources(sources)}\n{note}\nQUESTION: {state['question']}"
    answer = deps.llm.generate(load_prompt("answer_system"), prompt, 900)
    return {"answer": answer, "rewrites": state.get("rewrites", 0) + (1 if state.get("citation_checks") else 0)}
```

**`verify.py`**

```python
from app.agents.state import AgentState, Deps
from app.verification.citation_verifier import verify_citations


def verify_node(state: AgentState, deps: Deps) -> dict:
    checks = verify_citations(deps.llm, state["answer"], state["evidence"][:10])
    accuracy = sum(c.supported for c in checks) / len(checks) if checks else 1.0
    return {"citation_checks": [c.model_dump() for c in checks], "citation_accuracy": accuracy}
```

**`gaps.py`** — uses the graph from Phase 9

Add to `graph_repository.py`:

```python
UNTESTED_SQL = text("""
SELECT m.name AS method, d.name AS dataset
FROM entities m, entities d
WHERE m.kind = 'method' AND d.kind = 'dataset'
  AND (SELECT count(DISTINCT paper_id) FROM edges WHERE dst_id = m.id) >= 3
  AND (SELECT count(DISTINCT paper_id) FROM edges WHERE dst_id = d.id) >= 3
  AND NOT EXISTS (
        SELECT 1 FROM edges em JOIN edges ed ON em.paper_id = ed.paper_id
        WHERE em.dst_id = m.id AND ed.dst_id = d.id)
LIMIT :lim
""")


def untested_method_dataset_pairs(db: Session, limit: int = 30) -> list[dict]:
    return [dict(r._mapping) for r in db.execute(UNTESTED_SQL, {"lim": limit})]
```

`nodes/gaps.py`

```python
from app.agents.state import AgentState, Deps
from app.repositories import graph_repository

SYSTEM = (
    "You are given method/dataset pairs that no paper in the corpus has combined, plus a research question. "
    "Write up to 3 concrete, plausible research gaps relevant to the question. One sentence each, no numbering."
)


def gaps_node(state: AgentState, deps: Deps) -> dict:
    pairs = graph_repository.untested_method_dataset_pairs(deps.db)
    if not pairs:
        return {"gaps": []}
    listing = "\n".join(f"- {p['method']} x {p['dataset']}" for p in pairs)
    raw = deps.llm.generate(SYSTEM, f"QUESTION: {state['question']}\n\nUNTESTED PAIRS:\n{listing}", 300)
    return {"gaps": [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]}
```

### 10.4 Wire the graph

**`backend/app/agents/graph.py`**

```python
from functools import partial

from langgraph.graph import END, StateGraph

from app.agents.nodes.contradictions import contradictions_node
from app.agents.nodes.gaps import gaps_node
from app.agents.nodes.grade import grade_node
from app.agents.nodes.plan import plan_node
from app.agents.nodes.retrieve import retrieve_node
from app.agents.nodes.verify import verify_node
from app.agents.nodes.write import write_node
from app.agents.state import AgentState, Deps

MAX_SEARCH_LOOPS = 2
MIN_CITATION_ACCURACY = 0.8


def after_grade(state: AgentState) -> str:
    if state.get("sufficient") or state.get("iterations", 0) >= MAX_SEARCH_LOOPS:
        return "contradictions"
    return "retrieve"


def after_verify(state: AgentState) -> str:
    if state.get("citation_accuracy", 1.0) < MIN_CITATION_ACCURACY and state.get("rewrites", 0) < 2:
        return "write"
    return "gaps"


def build_agent(deps: Deps):
    g = StateGraph(AgentState)
    for name, fn in [
        ("plan", plan_node), ("retrieve", retrieve_node), ("grade", grade_node),
        ("contradictions", contradictions_node), ("write", write_node),
        ("verify", verify_node), ("gaps", gaps_node),
    ]:
        g.add_node(name, partial(fn, deps=deps))

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"retrieve": "retrieve", "contradictions": "contradictions"})
    g.add_edge("contradictions", "write")
    g.add_edge("write", "verify")
    g.add_conditional_edges("verify", after_verify, {"write": "write", "gaps": "gaps"})
    g.add_edge("gaps", END)
    return g.compile()
```

> **Study this file.** It is the whole control flow of the agent in 40 lines. Change `MAX_SEARCH_LOOPS` and observe.

### 10.5 Persist runs and stream events

**`backend/app/models/agent_run.py`**

```python
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="running")      # running | done | failed
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentEvent(Base):
    __tablename__ = "agent_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    node: Mapped[str] = mapped_column(String(30))
    payload: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

Register + migrate.

**`backend/app/services/run_agent.py`**

```python
import uuid

from app.agents.graph import build_agent
from app.agents.state import Deps
from app.db.session import SessionLocal
from app.llm.factory import get_llm
from app.models.agent_run import AgentEvent, AgentRun


def _summary(node: str, delta: dict) -> dict:
    """Small, UI-friendly description of what a node did."""
    return {
        "plan": {"queries": delta.get("plan")},
        "retrieve": {"evidence_count": len(delta.get("evidence", []))},
        "grade": {"sufficient": delta.get("sufficient")},
        "contradictions": {"count": len(delta.get("contradictions", []))},
        "write": {"answer_preview": (delta.get("answer") or "")[:200]},
        "verify": {"citation_accuracy": delta.get("citation_accuracy")},
        "gaps": {"gaps": delta.get("gaps")},
    }.get(node, {})


def execute_run(run_id: uuid.UUID) -> None:
    db = SessionLocal()
    run = db.get(AgentRun, run_id)
    try:
        agent = build_agent(Deps(db=db, llm=get_llm()))
        final: dict = {}
        for update in agent.stream({"question": run.question}, stream_mode="updates"):
            for node, delta in update.items():
                db.add(AgentEvent(run_id=run_id, node=node, payload=_summary(node, delta)))
                db.commit()
                final.update(delta)
        run.result, run.status = final, "done"
    except Exception as exc:
        run.result, run.status = {"error": str(exc)}, "failed"
    finally:
        db.commit()
        db.close()
```

**Endpoints:**

`agent_run_create.py` — `POST /agent/runs` `{ "question": "..." }` → creates `AgentRun`, schedules `execute_run`, returns `{id}`.

`agent_run_get.py` — `GET /agent/runs/{id}` → status + final result.

`agent_run_events.py` — `GET /agent/runs/{id}/events` (SSE):

```python
import json
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.db.session import SessionLocal
from app.models.agent_run import AgentEvent, AgentRun

router = APIRouter()


def _stream(run_id: uuid.UUID):
    last_id = 0
    while True:
        with SessionLocal() as db:
            events = db.query(AgentEvent).filter(AgentEvent.run_id == run_id, AgentEvent.id > last_id).order_by(AgentEvent.id).all()
            for e in events:
                last_id = e.id
                yield f"data: {json.dumps({'node': e.node, 'payload': e.payload})}\n\n"
            status = db.get(AgentRun, run_id).status
        if status != "running" and not events:
            yield f"event: end\ndata: {json.dumps({'status': status})}\n\n"
            return
        time.sleep(0.5)


@router.get("/agent/runs/{run_id}/events", tags=["agent"])
def run_events(run_id: uuid.UUID):
    return StreamingResponse(_stream(run_id), media_type="text/event-stream")
```

### Verify

```bash
RUN=$(curl -s -X POST localhost:8000/api/v1/agent/runs -H "Content-Type: application/json" \
      -d '{"question":"How do hybrid retrieval methods compare with dense-only retrieval, and where do studies disagree?"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
curl -N localhost:8000/api/v1/agent/runs/$RUN/events     # watch nodes fire live
curl localhost:8000/api/v1/agent/runs/$RUN               # final result
```

**Commit:** `phase 10: LangGraph agent`

---

# Phase 11 — Evaluation dashboard

### What and why
Without measurement, "it seems better" is a guess. You will build a **golden dataset** of questions with known relevant papers, run several **configurations**, and compare numbers. This is the phase that most clearly separates a production-style system from a demo.

### Technologies
| Tool | Purpose |
|------|---------|
| **Custom metric functions** (pure Python, unit-testable) | Recall@k, MRR, nDCG, hit rate |
| **LLM-as-judge** | Faithfulness and answer relevance |
| **PostgreSQL** | Store datasets, runs and per-question results |
| **MUI X Charts** | Dashboard visuals |
| *(Optional)* **Ragas** | Reference implementations to sanity-check your metrics |

### 11.1 Build the golden dataset (data gathering for evaluation)

1. **Synthetic first draft:** sample 150 random chunks. For each, ask the LLM: *"Write one specific question that can be answered only from this passage."* Save `(question, gold_paper_id, gold_chunk_id)`.
2. **Human review:** read each pair. Delete vague or unanswerable ones. Keep **50–100**. This manual step is the most valuable hour of the project.
3. **Add 10–20 hand-written hard questions** (acronyms, numbers, multi-paper comparisons).
4. **Add 10 unanswerable questions** (gold = none; correct behaviour is "not enough evidence").
5. Save as `data/eval/golden.jsonl`:

```json
{"question": "Which dataset was used to evaluate X?", "gold_paper_ids": ["<uuid>"], "answerable": true}
```

Tables (`models/eval.py`): `eval_questions(id, dataset_name, question, gold_paper_ids JSONB, answerable)`, `eval_runs(id, config JSONB, created_at, summary JSONB)`, `eval_results(id, run_id, question_id, retrieved_paper_ids JSONB, metrics JSONB, answer TEXT)`.

### 11.2 Metrics

**`backend/app/evaluation/retrieval_metrics.py`**

```python
import math


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant & set(retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(retrieved[:k])) / len(relevant) if relevant else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(retrieved[:k], start=1) if r in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
```

> Retrieval metrics operate on **paper ids** (de-duplicated, order preserved) so they stay meaningful even if you re-chunk.

**`backend/app/evaluation/generation_metrics.py`** — what each means and how to compute it

| Metric | Question it answers | How |
|--------|--------------------|-----|
| **Citation accuracy** | Of the cited sentences, what fraction does the cited text support? | Already returned by `/ask` (`citation_accuracy`) |
| **Citation coverage** | What fraction of factual sentences carry a citation? | `cited_sentences / total_sentences` using `split_cited_sentences` |
| **Faithfulness** | Is every claim grounded in the retrieved context (even uncited ones)? | LLM judge per sentence vs. *all* sources → fraction supported |
| **Answer relevance** | Does the answer address the question? | LLM judge score 1–5, divided by 5 |
| **Abstention accuracy** | On unanswerable questions, did it say "not enough evidence"? | String check on the fixed sentence |

```python
from app.verification.citation_parser import SENTENCE_SPLIT, split_cited_sentences


def citation_coverage(answer: str) -> float:
    sentences = [s for s in SENTENCE_SPLIT.split(answer) if len(s.split()) > 4]
    return len(split_cited_sentences(answer)) / len(sentences) if sentences else 0.0


def abstained(answer: str) -> bool:
    return "do not contain enough evidence" in answer.lower()
```

### 11.3 Runner

**Configurations to compare** (this table *is* your ablation study):

| Name | Retrieval |
|------|-----------|
| `bm25` | OpenSearch only |
| `vector` | pgvector only |
| `hybrid` | BM25 + vector + RRF |
| `hybrid_rerank` | + cross-encoder |
| `hybrid_rerank_graph` | + graph boost |

**`backend/app/evaluation/runner.py`** (sketch of the loop)

```python
RETRIEVERS = {
    "bm25": lambda db, q, k: bm25_repository.search_bm25(q, k),
    "vector": lambda db, q, k: vector_repository.search_vectors(db, embed_query(q), k),
    "hybrid": lambda db, q, k: hybrid_hits(db, q, k, use_rerank=False),
    "hybrid_rerank": lambda db, q, k: hybrid_hits(db, q, k, use_rerank=True),
}

def unique_paper_ids(hits):                        # keep order, drop duplicates
    return list(dict.fromkeys(h["paper_id"] for h in hits))

def run_retrieval_eval(db, config_name, questions, k=8):
    retrieve = RETRIEVERS[config_name]
    rows = []
    for q in questions:
        if not q.answerable:
            continue
        ids = unique_paper_ids(retrieve(db, q.question, k))
        rel = set(q.gold_paper_ids)
        rows.append({"hit": hit_at_k(ids, rel, k), "recall": recall_at_k(ids, rel, k),
                     "mrr": mrr(ids, rel), "ndcg": ndcg_at_k(ids, rel, k)})
    return {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]} if rows else {}
```

Generation evaluation runs `/ask` logic per question (slower and costs LLM calls — start with 20 questions).

### 11.4 Endpoints (one file each)

| File | Route |
|------|-------|
| `eval_dataset_import.py` | `POST /eval/dataset/import` (upload JSONL) |
| `eval_dataset_list.py` | `GET /eval/dataset` |
| `eval_run_create.py` | `POST /eval/runs` `{ "configs": ["bm25","vector","hybrid","hybrid_rerank"], "k": 8, "include_generation": false }` |
| `eval_run_list.py` | `GET /eval/runs` |
| `eval_run_get.py` | `GET /eval/runs/{id}` (per-question results, for drill-down) |
| `eval_summary.py` | `GET /eval/summary` (latest summary per config for charts) |

### 11.5 Dashboard components (frontend)

```
frontend/src/components/evaluation/
  ConfigComparisonChart.tsx   # BarChart: Recall@k, MRR, nDCG per config
  MetricCards.tsx             # big-number cards for the latest run
  RunsTable.tsx               # history of runs
  QuestionDrilldown.tsx       # failed questions: what was retrieved vs gold
```

**`ConfigComparisonChart.tsx`**

```tsx
"use client";
import { BarChart } from "@mui/x-charts/BarChart";

interface Row { config: string; recall: number; mrr: number; ndcg: number }

export default function ConfigComparisonChart({ rows }: { rows: Row[] }) {
  return (
    <BarChart
      height={340}
      xAxis={[{ scaleType: "band", data: rows.map((r) => r.config) }]}
      series={[
        { label: "Recall@8", data: rows.map((r) => r.recall) },
        { label: "MRR", data: rows.map((r) => r.mrr) },
        { label: "nDCG@8", data: rows.map((r) => r.ndcg) },
      ]}
    />
  );
}
```

### Verify and learn
Run all four configs. Expect hybrid ≥ either single method and reranking to improve MRR/nDCG most. If your results **disagree**, that's interesting: inspect `QuestionDrilldown` for queries where the winner lost. Write your conclusion (with numbers) in `LEARNING_LOG.md` and in a "Results" section of this README — this is what you show in interviews.

**Commit:** `phase 11: evaluation`

---

# Phase 12 — Frontend pages

Build one page per backend capability. All follow the pattern: **types → api file → component → page**.

| Page | Route | Components | Backend |
|------|-------|-----------|---------|
| Papers | `/papers` | `PapersTable`, `CollectDialog`, `UploadButton`, `IngestButton` | `/papers`, `/collect/arxiv`, `/papers/upload`, `/ingest` |
| Search | `/search` | `SearchBar`, `MethodToggle`, `HitCard` | `/search/*` |
| Ask | `/ask` | `QuestionForm`, `AnswerView`, `CitationChip`, `SourcesPanel`, `CitationChecks` | `/ask` |
| Agent | `/agent` | `AgentRunForm`, `AgentTimeline`, `ContradictionsList`, `GapsList` | `/agent/runs*` |
| Graph | `/graph` | `EntitySearch`, `GraphCanvas` | `/graph/*` |
| Evaluation | `/evaluation` | see Phase 11 | `/eval/*` |

### 12.1 Search page building blocks

**`frontend/src/types/search.ts`**

```ts
export interface SearchHit {
  chunk_id: string; paper_id: string; paper_title: string;
  section: string | null; page: number | null; text: string; score: number;
}
export interface SearchResponse { method: string; hits: SearchHit[] }
export type SearchMethod = "bm25" | "vector" | "hybrid";
```

**`frontend/src/lib/api/search.ts`**

```ts
import { apiFetch } from "./client";
import type { SearchMethod, SearchResponse } from "@/types/search";

export const search = (method: SearchMethod, query: string, k = 10) =>
  apiFetch<SearchResponse>(`/search/${method}`, { method: "POST", body: JSON.stringify({ query, k }) });
```

**`frontend/src/components/search/HitCard.tsx`**

```tsx
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import type { SearchHit } from "@/types/search";

export default function HitCard({ hit }: { hit: SearchHit }) {
  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="subtitle1" fontWeight={600}>{hit.paper_title}</Typography>
        <Stack direction="row" spacing={1} sx={{ my: 1 }}>
          {hit.section && <Chip size="small" label={hit.section} />}
          <Chip size="small" color="primary" label={`score ${hit.score.toFixed(3)}`} />
        </Stack>
        <Typography variant="body2" color="text.secondary">{hit.text.slice(0, 500)}…</Typography>
      </CardContent>
    </Card>
  );
}
```

### 12.2 Ask page — rendering citations

Turn `[1]` markers into clickable chips that scroll to the source.

**`frontend/src/components/ask/AnswerView.tsx`**

```tsx
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";

export default function AnswerView({ answer, onCite }: { answer: string; onCite: (n: number) => void }) {
  const parts = answer.split(/(\[\d+\])/g);              // keeps the markers
  return (
    <Typography component="div" sx={{ lineHeight: 1.9 }}>
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/);
        return m ? (
          <Chip key={i} size="small" label={m[1]} color="primary" variant="outlined"
                onClick={() => onCite(Number(m[1]))} sx={{ mx: 0.3 }} />
        ) : (
          <span key={i}>{part}</span>
        );
      })}
    </Typography>
  );
}
```

Color unsupported sentences red in `CitationChecks` using `citation_checks[].supported`. Showing the verification result to the user is a feature, not a debug tool.

### 12.3 Agent timeline with SSE

**`frontend/src/components/agent/useAgentEvents.ts`**

```ts
"use client";
import { useEffect, useState } from "react";

export interface AgentEvent { node: string; payload: Record<string, unknown> }

export function useAgentEvents(runId: string | null) {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!runId) return;
    setEvents([]); setDone(false);
    const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";
    const source = new EventSource(`${base}/agent/runs/${runId}/events`);
    source.onmessage = (e) => setEvents((prev) => [...prev, JSON.parse(e.data)]);
    source.addEventListener("end", () => { setDone(true); source.close(); });
    source.onerror = () => source.close();
    return () => source.close();
  }, [runId]);

  return { events, done };
}
```

Render with MUI `Stepper` or `Timeline`-style `List`: one row per event, icon per node, short payload summary.

### 12.4 Graph page
Install `react-force-graph-2d`, load it with `next/dynamic` and `{ ssr: false }` (it needs the browser). Fetch `/graph/neighbors/{id}?depth=2`, map edges to `{source, target}` links, color nodes by `kind`.

**Commit:** `phase 12: frontend pages`

---

# Phase 13 — Testing, CI, hardening

### Technologies
**pytest**, **FastAPI TestClient**, **ruff** (lint/format), **GitHub Actions**, **multi-stage Docker builds**.

### 13.1 Tests worth writing first
Pure functions are easiest and most valuable:

**`backend/tests/test_rrf.py`**

```python
from app.retrieval.rrf import reciprocal_rank_fusion


def test_item_in_both_lists_wins():
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    b = [{"chunk_id": "z"}, {"chunk_id": "x"}]
    assert reciprocal_rank_fusion([a, b])[0]["chunk_id"] == "x"
```

**`backend/tests/test_metrics.py`**

```python
from app.evaluation.retrieval_metrics import mrr, recall_at_k


def test_mrr_first_hit_at_rank_two():
    assert mrr(["a", "b", "c"], {"b"}) == 0.5


def test_recall_at_k():
    assert recall_at_k(["a", "b"], {"a", "z"}, 2) == 0.5
```

Also test: `split_cited_sentences`, `tei_parser` on a saved small TEI sample, `split_words` overlap, and `/health`.

```bash
# requirements-dev.txt: pytest, ruff
docker compose exec backend pytest -q
docker compose exec backend ruff check .
```

### 13.2 CI

**`.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]
jobs:
  backend:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: backend } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff pytest pydantic-settings
      - run: ruff check .
      - run: pytest tests/unit -q          # keep CI tests free of model downloads / DB
  frontend:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: frontend } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - run: npm ci
      - run: npm run lint
      - run: npm run build
```

### 13.3 Hardening checklist
- [ ] Limit upload size (reject PDFs > 30 MB) and verify the PDF magic bytes (`%PDF`).
- [ ] Add an API key header or login before exposing the API publicly.
- [ ] Rate-limit `/ask` and `/agent/runs` (LLM calls cost money) — e.g. `slowapi`.
- [ ] Never log API keys or full prompts containing private documents.
- [ ] Remove `--reload` and source-code volume mounts in production compose.
- [ ] Enable OpenSearch security (remove `DISABLE_SECURITY_PLUGIN`) outside local dev.
- [ ] Replace `BackgroundTasks` with a real queue (Celery/RQ/ARQ + Redis) if ingestion must survive restarts.
- [ ] Add structured logging with request IDs; log latency per pipeline stage (retrieve / rerank / generate / verify).
- [ ] Production frontend image (multi-stage):

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app ./
EXPOSE 3000
CMD ["npm", "run", "start"]
```

**Commit:** `phase 13: tests, CI, hardening`

---

# Endpoint catalogue

Base URL: `http://localhost:8000/api/v1` · Interactive docs: `/docs`

| Method | Path | File | Phase | Purpose |
|--------|------|------|-------|---------|
| GET | `/health` | `health.py` | 2 | Liveness + DB check |
| GET | `/papers` | `papers_list.py` | 2 | Paginated list |
| GET | `/papers/{id}` | `papers_get.py` | 2 | One paper |
| POST | `/papers/upload` | `papers_upload.py` | 4 | Upload a PDF |
| DELETE | `/papers/{id}` | `papers_delete.py` | 6 | Delete paper, chunks, index entries |
| GET | `/papers/{id}/chunks` | `papers_chunks.py` | 6 | Inspect chunks (debugging) |
| GET | `/papers/{id}/references` | `papers_references.py` | 5 | Bibliography |
| POST | `/collect/arxiv` | `collect_arxiv.py` | 4 | Fetch metadata + PDFs |
| POST | `/ingest/{paper_id}` | `ingest_paper.py` | 6 | Parse → chunk → embed → index one paper |
| POST | `/ingest` | `ingest_all.py` | 6 | Ingest all downloaded papers |
| POST | `/search/bm25` | `search_bm25.py` | 7 | Keyword search |
| POST | `/search/vector` | `search_vector.py` | 6 | Semantic search |
| POST | `/search/hybrid` | `search_hybrid.py` | 7 | BM25 + vector + RRF (+ rerank) |
| POST | `/ask` | `ask.py` | 8 | Cited answer + verification |
| POST | `/compare` | `compare.py` | 8 | Cross-paper comparison table |
| POST | `/graph/build/{paper_id}` | `graph_build.py` | 9 | Extract graph for a paper |
| POST | `/graph/build` | `graph_build_all.py` | 9 | Build for all indexed papers |
| GET | `/graph/entities` | `graph_search_entities.py` | 9 | Search entities |
| GET | `/graph/neighbors/{entity_id}` | `graph_neighbors.py` | 9 | Subgraph |
| GET | `/graph/entities/{id}/papers` | `graph_papers_by_entity.py` | 9 | Papers by entity |
| POST | `/gaps` | `gaps.py` | 10 | Research gaps from graph |
| POST | `/agent/runs` | `agent_run_create.py` | 10 | Start agent run |
| GET | `/agent/runs/{id}` | `agent_run_get.py` | 10 | Run status + result |
| GET | `/agent/runs/{id}/events` | `agent_run_events.py` | 10 | SSE event stream |
| POST | `/eval/dataset/import` | `eval_dataset_import.py` | 11 | Upload golden JSONL |
| GET | `/eval/dataset` | `eval_dataset_list.py` | 11 | List questions |
| POST | `/eval/runs` | `eval_run_create.py` | 11 | Run an evaluation |
| GET | `/eval/runs` | `eval_run_list.py` | 11 | History |
| GET | `/eval/runs/{id}` | `eval_run_get.py` | 11 | Per-question detail |
| GET | `/eval/summary` | `eval_summary.py` | 11 | Chart data |

---

# Database schema

| Table | Key columns | Phase |
|-------|-------------|-------|
| `papers` | id, source, source_id (unique pair), title, abstract, doi, year, pdf_url, pdf_path, status | 2 |
| `authors` / `paper_authors` | author name; many-to-many link | 4 |
| `chunks` | id, paper_id, chunk_index, chunk_type, section, page, text, **embedding vector(384)** (HNSW) | 5–6 |
| `references` | id, paper_id, title, year, doi | 5 |
| `entities` | id, kind, name (unique pair), ref_id | 9 |
| `edges` | id, src_id, dst_id, relation, paper_id, evidence, weight | 9 |
| `agent_runs` / `agent_events` | question, status, result JSONB; per-node events | 10 |
| `eval_questions` / `eval_runs` / `eval_results` | golden set; runs with config + summary; per-question metrics | 11 |

Paper `status` lifecycle: `metadata_only → downloaded → parsed → indexed` (or `failed`).

---

# Troubleshooting

| Symptom | Likely cause / fix |
|---------|-------------------|
| Frontend shows network/CORS error | `CORS_ORIGINS` missing your frontend URL; restart backend |
| `connection refused` to Postgres | Backend started before DB healthy; check `depends_on` + health check |
| Alembic sees no changes | Model not imported in `app/models/__init__.py` |
| `type "vector" does not exist` | `init.sql` didn't run (volume already existed). Run `CREATE EXTENSION vector;` manually or `docker compose down -v` |
| First `/search/vector` is very slow | Model downloading/loading once; use the `hf_cache` volume |
| OpenSearch exits immediately | Not enough memory (give Docker ≥ 4 GB) or set `vm.max_map_count` on Linux |
| GROBID returns 503 | Busy — it processes few PDFs in parallel; retry with backoff |
| Parsed sections empty | Scanned PDF (image only). Needs OCR (e.g. `ocrmypdf`) |
| LLM JSON parse failures | Lower `max_tokens` issues or chatty model; keep `parse_json` fallbacks and log raw replies |
| arXiv returns 429/403 | You exceeded 1 request / 3 seconds — keep the sleeps |
| MUI hydration mismatch | Missing `AppRouterCacheProvider`, or wrong `v1x-appRouter` import for your Next version |

---

# Stretch goals and resources

**Stretch goals (in rough order of value)**
1. Streaming `/ask` answers (SSE) for a ChatGPT-style feel.
2. Section-aware filters in search (only "Results" sections, year ranges, authors).
3. Query rewriting / HyDE (generate a hypothetical answer, embed that).
4. Fine-tune the embedding model on your synthetic (question, chunk) pairs and show the eval delta.
5. Neo4j for the graph, plus entity de-duplication using embedding similarity.
6. Contradiction detection with an NLI model; show a "conflict matrix" in the UI.
7. Figure understanding: send figure images to a vision model and index the descriptions.
8. Scheduled collection (weekly arXiv pull) with a Celery beat job.
9. Multi-user accounts with per-user collections.

**Reading list (learn the *why*)**
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (RAG paper)
- Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*
- Muennighoff et al., *MTEB: Massive Text Embedding Benchmark*
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
- Official docs: FastAPI, SQLAlchemy 2.0, pgvector, OpenSearch, LangGraph, Next.js App Router, MUI

---

## Final "definition of done"

- [ ] `docker compose up` starts Postgres, OpenSearch, GROBID, backend, frontend with no manual steps beyond `.env`
- [ ] ≥ 300 papers ingested from one coherent domain, with section-aware chunks, tables, captions, references
- [ ] Four retrieval configurations compared on a reviewed golden set, with numbers in the README
- [ ] `/ask` returns citations; verification shows citation accuracy for every answer
- [ ] Agent run visible step by step in the UI, including contradictions and gaps
- [ ] Unit tests for pure logic + CI green
- [ ] `LEARNING_LOG.md` has an entry for every phase
- [ ] A short demo video / screenshots in this README

> **Build order reminder:** Phases 0–3 are scaffolding. The real value starts at Phase 4. Do not skip evaluation (Phase 11) — it is the proof that your system works.

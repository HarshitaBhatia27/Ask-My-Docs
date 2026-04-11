---
title: AskMyDocs
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# AskMyDocs 📚
**Production-grade RAG system** — upload any PDF, ask questions in natural language, get cited answers grounded strictly in the document. Zero hallucination by design.

---

## Architecture

```
PDF Upload
    │
    ▼
PyPDFLoader → RecursiveCharacterTextSplitter (1000 chars / 400 overlap)
    │
    ▼
BAAI/bge-small-en Embeddings  ──→  ChromaDB (persistent vector store)
    │
    ▼
Query
    │
    ├─── Vector Search (ChromaDB, k=10) ──┐
    │                                      ├──→ Deduplicated Candidate Pool
    └─── BM25 Keyword Search (k=2)  ──────┘
                                           │
                                           ▼
                              CrossEncoder Reranker (ms-marco-MiniLM-L-6-v2)
                              Scores all (query, chunk) pairs → top 4
                                           │
                                           ▼
                              Groq Llama 3.3 70B
                              Context-only prompt + mandatory page citations
                                           │
                                           ▼
                              Cited Answer  (or "Not found in documents")
```

---

## Design Decisions & Why

### Why Hybrid Retrieval (BM25 + Vector)?
Vector search (dense retrieval) finds semantically similar chunks — useful for conceptual or paraphrased queries. BM25 (sparse keyword retrieval) matches exact terms — useful for specific names, code, or acronyms. Neither alone covers all query types. Combining them gives the system the best of both worlds: semantic understanding *and* exact-term recall.

### Why Cross-Encoder Reranking?
The retrieval step uses fast bi-encoders (embeddings pre-computed once). These are efficient but approximate — they score a query and chunk independently. A cross-encoder sees the query and chunk *together*, computing a richer relevance score. It's too slow to run on every chunk in the database, but after retrieval narrows candidates to 10, running a cross-encoder on those 10 is cheap and significantly improves precision.

### Why 1000-char chunks with 400-char overlap?
Chunks need to be large enough to contain a self-contained idea (too small = loss of context), but small enough to stay focused (too large = noise dilutes relevance scores). 1000 chars ≈ one paragraph. The 400-char overlap ensures a sentence that falls at a chunk boundary is captured in both adjacent chunks, preventing answers from falling through the cracks.

### Why BAAI/bge-small-en for embeddings?
Runs fully local — no API calls, no cost, no rate limits. BGE models consistently rank at the top of the MTEB leaderboard for retrieval tasks at the small model size. Using a cloud embedding API would add latency and cost per query; local embeddings add a one-time model load cost instead.

### Why Groq + Llama 3.3 70B?
Groq's inference hardware (LPUs) delivers very low latency on large models — Llama 3.3 70B responds in 1–3 seconds, comparable to much smaller hosted models. Using a 70B model improves answer quality and instruction-following (strict context-only + citation prompts) over smaller alternatives.

### Why Context-Only Prompting?
The system prompt explicitly instructs the LLM to answer only from provided context and to say "Not found in documents" if it cannot. This eliminates hallucination at the cost of occasionally returning no answer — a deliberate, correct tradeoff for a document Q&A use case.

### Why FastAPI + Streamlit as separate services?
Decoupling the backend (FastAPI) from the frontend (Streamlit) means the API can be tested, replaced, or called from other clients independently. In Docker Compose, Streamlit waits for FastAPI to pass a health check before starting — this prevents race conditions on boot.

---

## Evaluation (RAGAS)

Evaluated on a 9.8MB Algorithms textbook PDF with a set of factual questions requiring specific retrieval.

| Metric | Score | What it measures |
|---|---|---|
| **Answer Relevance** | **0.97** | Are answers actually addressing the question asked? |
| **Faithfulness** | **0.88** | Does the answer stay within the retrieved context (no hallucination)? |
| **Context Precision** | **0.86** | Is the retrieved context relevant to the question (low noise)? |
| **Context Recall** | **0.50** | Is all relevant information from the document being retrieved? |

**Interpretation:** Answer quality is high — when the system retrieves the right context, answers are relevant (0.97) and faithful (0.88). Context Precision (0.86) confirms the retrieved chunks are mostly relevant. Context Recall (0.50) is the known weak point: roughly half of relevant document passages are not being surfaced. This is a retrieval coverage problem, not a generation problem.

**Known cause & planned fix:** A context recall of 0.50 typically means the chunk containing the answer exists but ranks below top-10 during retrieval, so the reranker never sees it. Planned improvements: (1) increase retrieval candidate pool from k=10 to k=20 before reranking, (2) experiment with larger chunk size (1500 chars) to reduce fragmentation of long explanations, (3) evaluate adding a parent-document retriever to retrieve full sections when a child chunk matches.

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Groq API — Llama 3.3 70B | Ultra-low latency inference |
| Embeddings | BAAI/bge-small-en | Local, free, top MTEB retrieval score |
| Vector Store | ChromaDB (persistent) | Lightweight, no infra required |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder trained on MS MARCO retrieval |
| Keyword Search | BM25 (rank_bm25) | Exact-term matching to complement vector search |
| API | FastAPI | Async, typed, auto-docs at /docs |
| Frontend | Streamlit | Fast to build; session-state prevents redundant indexing |
| Containerisation | Docker + Docker Compose | Reproducible, service-isolated deployments |
| CI/CD | GitHub Actions + GHCR | Automated tests, Docker build, and image push on merge |
| Evaluation | RAGAS | RAG-specific metrics: faithfulness, recall, precision |

---

## Setup

```bash
git clone https://github.com/HarshitaBhatia27/askmydocs
cd askmydocs
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

---

## Run

### Option 1 — Docker Compose (recommended)
```bash
docker-compose up --build
```
FastAPI starts first, Streamlit waits for it to be healthy, then both are available:
- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### Option 2 — Local
```bash
# Terminal 1 — start API
uvicorn api.main:app --reload

# Terminal 2 — start UI
streamlit run app.py
```

---

## Project Structure

```
├── src/
│   ├── chat_model.py    # LLM (Groq) + embeddings (BGE) initialisation
│   ├── indexer.py       # PDF loading, chunking, ChromaDB ingestion
│   └── retriever.py     # Hybrid search, cross-encoder reranking, answer generation
├── api/
│   └── main.py          # FastAPI: /upload and /ask endpoints
├── tests/
│   └── test_api.py      # Unit tests with mocked dependencies
├── app.py               # Streamlit chat interface
├── evaluation.py        # RAGAS evaluation script
├── Dockerfile           # Single image for both services
├── docker-compose.yml   # FastAPI + Streamlit with health-check dependency
├── .github/
│   └── workflows/
│       └── ci.yml       # CI: test → build → push to GHCR on merge to main
└── requirements.txt
```

---

## CI/CD Pipeline

On every push or PR to `main`:
1. **Test job** — installs lightweight deps, runs `pytest tests/ -v` with mocked heavy dependencies
2. **Build & push job** (main branch only, after tests pass) — builds Docker image with layer caching, pushes to GitHub Container Registry tagged `:latest` and `:<commit-sha>`

The commit SHA tag enables rollbacks to any previous build.

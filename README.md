# AskMyDocs 📚

A production-grade RAG (Retrieval-Augmented Generation) system that lets you upload any PDF and ask questions about it in natural language. Answers are grounded strictly in the document with page citations — zero hallucination.

## Demo
Upload a PDF → Ask questions → Get cited answers from your document

## Architecture
```
PDF Upload → Chunking → Embeddings → ChromaDB
     ↓
Query → Hybrid Search (BM25 + Vector) → Cross-Encoder Reranking → Groq LLM → Cited Answer
```

## What Makes This Production-Grade
- **Hybrid Retrieval** — BM25 keyword search + vector similarity search combined. Pure vector search misses exact terms; pure BM25 misses semantic meaning. Both together cover all query types.
- **Cross-Encoder Reranking** — After retrieving 10 candidates, a cross-encoder rescores each (query, chunk) pair and selects the best 4. More accurate than bi-encoder retrieval alone.
- **Citation Enforcement** — Every answer cites page numbers. The LLM is prompted to only use provided context — if it can't cite it, it won't say it.
- **Hallucination Prevention** — Explicit "Not found in documents" fallback prevents the LLM from guessing.
- **Session-Aware UI** — Streamlit session state ensures the document is only indexed once per session regardless of UI reruns.

## Tech Stack
| Component | Technology |
|---|---|
| LLM | Groq API — Llama 3.3 70B |
| Embeddings | BAAI/bge-small-en (local, free) |
| Vector Store | ChromaDB (persistent, local) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| API | FastAPI |
| Frontend | Streamlit |

## Setup
```bash
git clone https://github.com/yourusername/askmydocs
cd askmydocs
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

## Run
```bash
# Terminal 1 — start API
uvicorn api.main:app --reload

# Terminal 2 — start UI
streamlit run app.py
```

Open `localhost:8501`, upload a PDF, start asking questions.

## Project Structure
```
├── src/
│   ├── chat_model.py    # LLM + embeddings initialization
│   ├── indexer.py       # PDF loading, chunking, ChromaDB storage
│   └── retriever.py     # Hybrid search, reranking, answer generation
├── api/
│   └── main.py          # FastAPI endpoints (/upload, /ask)
├── app.py               # Streamlit chat interface
└── requirements.txt
```
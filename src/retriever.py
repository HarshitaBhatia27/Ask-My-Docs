import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from src.chat_model import get_llm, get_embeddings

DB_PATH = os.getenv("CHROMA_PATH", "/tmp/local_chroma_db")

# initialize lazily
_model = None
_embeddings = None
_reranker = None
vector_store = None
bm25_retriever = None

def get_model():
    global _model
    if _model is None:
        _model = get_llm()
    return _model

def get_embeddings_cached():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def reload_vector_store():
    global vector_store, bm25_retriever
    vector_store = Chroma(
        embedding_function=get_embeddings_cached(),
        persist_directory=DB_PATH
    )
    raw = vector_store.get()
    bm25_retriever = BM25Retriever.from_texts(raw['documents'], k=2)
    print("Retriever reloaded.")

def hybrid_search(query: str, k: int = 10):
    vector_results = vector_store.similarity_search(query, k=k)
    bm25_results = bm25_retriever.invoke(query)
    seen = set()
    combined = []
    for doc in vector_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
    return combined[:k]

def rerank(query: str, docs: list, top_k: int = 4):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]

def ask(question: str) -> str:
    if vector_store is None:
        return "Please upload a document first."
    docs = hybrid_search(question, k=10)
    best_docs = rerank(question, docs, top_k=4)
    context = "\n\n".join([
        f"[Page {doc.metadata.get('page', i+1)}]: {doc.page_content}"
        for i, doc in enumerate(best_docs)
    ])
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a technical document.
Use the context below to give a clear, direct answer.
Cite page numbers like [Page 5] for every claim.
If the answer is not in the context, say "Not found in documents."

Context:
{context}

Question: {question}

Answer:""")
    chain = prompt | get_model()
    response = chain.invoke({"context": context, "question": question})
    return response.content
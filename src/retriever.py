import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from src.chat_model import get_llm, get_embeddings

DB_PATH = os.getenv("CHROMA_PATH", "/tmp/local_chroma_db")

model = get_llm()
embeddings = get_embeddings()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

vector_store = None
bm25_retriever = None

def reload_vector_store():
    global vector_store, bm25_retriever
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    raw = vector_store.get()
    bm25_retriever = BM25Retriever.from_texts(raw['documents'], k=2)
    print("Retriever reloaded with new document.")

if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    reload_vector_store()

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
    scores = reranker.predict(pairs)
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

    # prompt and chain MUST be inside ask()
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a technical document.

Use the context below to give a clear, direct answer.
If the context contains a definition, state it clearly first then elaborate.
Cite page numbers like [Page 5] for every claim.
If the answer is not in the context, say "Not found in documents."

Context:
{context}

Question: {question}

Answer:""")

    chain = prompt | model
    response = chain.invoke({"context": context, "question": question})
    return response.content  # ← this was missing entirely
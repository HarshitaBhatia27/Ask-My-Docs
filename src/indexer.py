import os, sys, shutil, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.retriever import get_embeddings_cached as get_embeddings

DB_PATH = os.getenv("CHROMA_PATH", "/tmp/local_chroma_db")

def build_vector_store_from_file(file_path: str):
    # Completely remove and recreate — avoids ChromaDB schema mismatch errors
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("Cleared old vector store")
    os.makedirs(DB_PATH, exist_ok=True)
    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=50    # large chunks + tiny overlap = fewest chunks possible
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Insert in batches of 500 to avoid memory spikes on HF Spaces free CPU
    batch_size = 500
    embeddings = get_embeddings()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if i == 0:
            db = Chroma.from_documents(batch, embedding=embeddings, persist_directory=DB_PATH)
        else:
            db.add_documents(batch)
        print(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
    print("Vector store ready.")
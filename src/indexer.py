import os, sys, shutil, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.chat_model import get_embeddings

DB_PATH = "./local_chroma_db"

def build_vector_store_from_file(file_path: str):
    if os.path.exists(DB_PATH):
        # Delete contents, not the directory itself (volume mount stays intact)
        for item in os.listdir(DB_PATH):
            item_path = os.path.join(DB_PATH, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print("Cleared old vector store")
    else:
        os.makedirs(DB_PATH, exist_ok=True)
    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=DB_PATH
    )
    print("Vector store ready.")
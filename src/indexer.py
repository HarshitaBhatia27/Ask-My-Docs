import os, sys, shutil, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.chat_model import get_embeddings

DB_PATH = "./local_chroma_db"

def build_vector_store_from_file(file_path: str):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        time.sleep(1)  # wait for OS to release file lock
        print("Cleared old vector store")

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
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    print(f"DEBUG: GROQ_API_KEY present = {bool(api_key)}, length = {len(api_key) if api_key else 0}")
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        encode_kwargs={
            "batch_size": 64,           # process 64 chunks at once instead of 1
            "normalize_embeddings": True # BGE models are trained with normalization
        }
    )
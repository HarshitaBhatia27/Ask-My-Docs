import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
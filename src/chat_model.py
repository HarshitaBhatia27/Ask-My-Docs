import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
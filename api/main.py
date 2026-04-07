from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import build_vector_store_from_file
from src.retriever import ask, reload_vector_store

app = FastAPI(title="AskMyDocs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "AskMyDocs API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # ensure data/ directory exists (may not exist in a fresh container)
    os.makedirs("data", exist_ok=True)

    # save uploaded file temporarily
    temp_path = f"data/uploaded_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # wipes old DB and builds fresh
    build_vector_store_from_file(temp_path)
    
    # reload retriever with new vector store
    reload_vector_store()
    
    return {"status": "indexed", "filename": file.filename}

@app.post("/ask")
def ask_question(query: Query):
    answer = ask(query.question)
    return {"question": query.question, "answer": answer}
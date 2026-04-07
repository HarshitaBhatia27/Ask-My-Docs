import sys
import os
from unittest.mock import patch, MagicMock

# CI only installs lightweight test deps (fastapi, pytest, httpx, pydantic).
# These packages are not installed in CI because they are too large (~2GB).
# Inserting MagicMock() into sys.modules fools Python's import system into thinking they are already loaded, so the import chain succeeds without
# actually needing the real packages. The @patch decorators in each test
# then replace the specific functions with controlled mocks.
_MOCK_MODULES = [
    "dotenv",
    "langchain_groq",
    "langchain_huggingface",
    "langchain_chroma",
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_community.retrievers",
    "langchain_core",
    "langchain_core.prompts",
    "langchain_text_splitters",
    "sentence_transformers",
    "chromadb",
    "rank_bm25",
    "torch",
]
for _mod in _MOCK_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "AskMyDocs API is running"


@patch("api.main.build_vector_store_from_file")
@patch("api.main.reload_vector_store")
@patch("api.main.shutil.copyfileobj")   # prevent actual file I/O
@patch("builtins.open", MagicMock())    
def test_upload_pdf(mock_copy, mock_reload, mock_build):
    mock_build.return_value = None
    mock_reload.return_value = None
    mock_copy.return_value = None

    dummy_pdf = b"%PDF-1.4 fake pdf content"
    response = client.post(
        "/upload",
        files={"file": ("test.pdf", dummy_pdf, "application/pdf")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    assert response.json()["filename"] == "test.pdf"
    mock_build.assert_called_once()
    mock_reload.assert_called_once()


@patch("api.main.ask")
def test_ask_question(mock_ask):
    mock_ask.return_value = "This is a test answer [Page 1]."
    response = client.post("/ask", json={"question": "What is dropout?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert response.json()["question"] == "What is dropout?"


@patch("api.main.ask")
def test_ask_no_document(mock_ask):
    mock_ask.return_value = "Please upload a document first."
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 200
    assert "answer" in response.json()


def test_ask_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 200

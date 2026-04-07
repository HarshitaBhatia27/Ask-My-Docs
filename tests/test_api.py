import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "AskMyDocs API is running"

@patch("api.main.build_vector_store_from_file")
@patch("api.main.reload_vector_store")
def test_upload_pdf(mock_reload, mock_build):
    mock_build.return_value = None
    mock_reload.return_value = None

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
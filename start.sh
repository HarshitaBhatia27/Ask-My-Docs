#!/bin/bash
# Starts both services inside a single container.
# FastAPI runs on port 8000 (internal only).
# Streamlit runs on port 7860 (exposed — HF Spaces default).

# Start FastAPI in the background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Give FastAPI a moment to be ready before Streamlit tries to call it
sleep 3

# Start Streamlit in the foreground (keeps the container alive)
streamlit run app.py --server.port 7860 --server.address 0.0.0.0

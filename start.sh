#!/bin/bash
export GROQ_API_KEY=$(echo $GROQ_API_KEY)
streamlit run app.py --server.port 7860 --server.address 0.0.0.0

#!/bin/bash
# HF Spaces runs behind a reverse proxy. Streamlit's XSRF protection
# doesn't play well with it and causes 403 on file uploads — disabling it fixes that.
export GROQ_API_KEY=$(echo $GROQ_API_KEY)

streamlit run app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.enableXsrfProtection false

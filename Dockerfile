FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data /tmp/local_chroma_db

# Pre-download both models at build time so they are baked into the image.
# Without this, the first PDF upload triggers a model download at runtime
# which is slow and can fail on restricted networks.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

RUN chmod +x start.sh

# Port 7860 is the default exposed port on Hugging Face Spaces
EXPOSE 7860

CMD ["./start.sh"]

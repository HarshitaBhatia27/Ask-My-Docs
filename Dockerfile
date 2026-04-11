FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data local_chroma_db

# Make the startup script executable
RUN chmod +x start.sh

# Port 7860 is the default exposed port on Hugging Face Spaces
EXPOSE 7860

CMD ["./start.sh"]

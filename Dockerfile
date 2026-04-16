FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e "." \
    && pip install --no-cache-dir \
        fastapi uvicorn sentence-transformers

COPY . .

ENV SOMA_BUNDLE_PATH=/app/data/memory
ENV SOMA_EMBED_MODEL=all-MiniLM-L6-v2

EXPOSE 8420

CMD ["uvicorn", "soma.serve:app", "--host", "0.0.0.0", "--port", "8420"]

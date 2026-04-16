FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Core + serve extras (fastapi/uvicorn/sentence-transformers) in one layer
# so image builds cleanly even when the source tree isn't yet copied.
RUN pip install --no-cache-dir -e ".[serve]"

COPY . .

# Default single-tenant bundle; SOMA_BUNDLES_DIR for /bundles/{name} routes.
ENV SOMA_BUNDLE_PATH=/app/data/memory \
    SOMA_BUNDLES_DIR=/app/data/bundles \
    SOMA_EMBED_MODEL=all-MiniLM-L6-v2

EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8420/health', timeout=2).read()" || exit 1

CMD ["uvicorn", "soma.serve:app", "--host", "0.0.0.0", "--port", "8420"]

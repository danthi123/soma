FROM python:3.12-slim

WORKDIR /app

# Pre-bake hugging-face cache location so the sbert snapshot we download
# at build time lives inside the image (not a volatile tmpfs layer).
ENV HF_HOME=/app/.cache/huggingface

# Layer cache: copy only the dependency-manifest + README (pyproject
# references it via `readme = "README.md"`) first so `pip install -e` is
# cached across source-only edits.
COPY pyproject.toml README.md ./
COPY src/ src/

# Core + serve extras (fastapi/uvicorn/sentence-transformers) in one layer.
RUN pip install --no-cache-dir -e ".[serve]"

# Pre-bake the default sbert model during the build so cold-starts don't
# pay a ~60 s model download on first /retrieve. Adds ~90 MB to the image
# but drops cold-start to ~8 s. Overridable at runtime via SOMA_EMBED_MODEL.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the repo (tests, configs, scripts, etc.) after the
# dependency layer so source-only changes don't invalidate the pip install.
COPY . .

# Default single-tenant bundle; SOMA_BUNDLES_DIR for /bundles/{name} routes.
ENV SOMA_BUNDLE_PATH=/app/data/memory \
    SOMA_BUNDLES_DIR=/app/data/bundles \
    SOMA_EMBED_MODEL=all-MiniLM-L6-v2 \
    PORT=8420

EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8420/health', timeout=2).read()" || exit 1

# Shell-form CMD so $PORT expands at runtime (Railway / Render / Fly each
# inject their own port; default 8420 for local `docker run`).
CMD ["sh", "-c", "uvicorn soma.serve:app --host 0.0.0.0 --port ${PORT:-8420}"]

# AlphaGenome × Aging Variant Study — Hugging Face Spaces (Docker SDK)
#
# Builds a FastAPI app serving:
#   - /                       static webapp landing page
#   - /api/healthz            liveness check
#   - /api/gallery            precomputed predictions
#   - /api/predict/{rsid}     live AlphaGenome inference
#   - /api/compare?a=&b=      side-by-side
#   - /docs/pipeline.html     project documentation
#
# The AlphaGenome API key is read from the ALPHAGENOME_API_KEY env var
# (set as a Space secret in the HF UI — never bake it into the image).

FROM python:3.10-slim

# System deps: build tools for pysam, libcurl for tabix-over-HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libcurl4-openssl-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libssl-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as a non-root user (uid 1000) — match that
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install Python deps first for cache-friendliness
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app source. The .dockerignore keeps the huge data/alphagenome_scores
# directory out of the build context.
COPY --chown=user:user . .

USER user

# Hugging Face Spaces expects the app on port 7860 by default, but our
# README frontmatter sets app_port: 8000 so we listen there for consistency
# with local dev.
EXPOSE 8000

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]

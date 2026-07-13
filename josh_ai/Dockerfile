# ---- Stage 1: build/install dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /build

# system deps needed only to build some python packages (removed in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: lean final image ----
FROM python:3.11-slim

WORKDIR /app

# copy only the installed python packages, not build tools
COPY --from=builder /install /usr/local

# copy only inference-time code and model artifacts
COPY deploy/ /app/

# no network access needed at runtime -- models are baked into the image
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
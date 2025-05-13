FROM python:3.10-slim

# System dependencies for hnswlib and other C++ extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HNSW_DIM=512 \
    HNSW_MAX_ELEMENTS=100000 \
    HNSW_SPACE=cosine \
    HNSW_EF_CONSTRUCTION=200 \
    HNSW_M=16 \
    HNSW_EF_SEARCH=50

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Build context is the repo root (see docker-compose.yml). Install
# dependencies first (best layer caching) then copy app code, ops
# scripts, and golden eval data.
COPY backend/pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY backend/app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

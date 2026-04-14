FROM python:3.13-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY . .

# ---- API server ----
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "medqa.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Streamlit frontend ----
FROM base AS frontend
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY ui ./ui

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "streamlit>=1.40.0" \
        "requests>=2.32.0"

# API base URL is injected at run time (env var).
ENV API_URL=http://localhost:8000

EXPOSE 8501
HEALTHCHECK NONE
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
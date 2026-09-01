FROM python:3.14-slim

WORKDIR /app

# System deps for building any wheels streamlit/fastapi might need incase they
# are not published for py3.14 yet. Kept minimal.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY api ./api
COPY models ./models
COPY data ./data
COPY README.md ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "fastapi>=0.115.0" \
        "uvicorn>=0.34.0" \
        "scikit-learn>=1.9.0" \
        "pandas>=3.0.5" \
        "joblib"

ENV PYTHONPATH=/app:/app/src

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
#!/usr/bin/env bash
# Run the CGPA prediction demo locally.
#
#   ./run_demo.sh [API_PORT] [UI_PORT]
#
# Starts the FastAPI model API and the Streamlit UI, then opens the UI.
# Default ports: API=8010 (8000 is often taken locally), UI=8501.

set -euo pipefail

API_PORT="${1:-8010}"
UI_PORT="${2:-8501}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || { echo "No .venv found — run: uv sync"; exit 1; }

echo "Starting model API on :$API_PORT ..."
PORT="$API_PORT" nohup "$PYTHON" -m api.main >/tmp/cgpa_api.log 2>&1 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

echo "Waiting for API /health ..."
for _ in $(seq 1 20); do
  "$PYTHON" -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:$API_PORT/health', timeout=2)" 2>/dev/null && break
  sleep 0.5
done

echo "Starting Streamlit UI on :$UI_PORT ... (open http://localhost:$UI_PORT)"
API_URL="http://localhost:$API_PORT" "$PYTHON" -m streamlit run ui/app.py \
  --server.port="$UI_PORT" --server.address=127.0.0.1

kill $API_PID 2>/dev/null || true
trap - EXIT
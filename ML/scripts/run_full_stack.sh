#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
FRONTEND_DIR="$PROJECT_DIR/frontend"
VENV_DIR="$PROJECT_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/uvicorn" ]]; then
  echo "Backend environment missing. Run: cd \"$PROJECT_DIR\" && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Frontend dependencies missing. Run: cd \"$FRONTEND_DIR\" && npm install"
  exit 1
fi

API_PID=""
WEB_PID=""

cleanup() {
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "$WEB_PID" ]] && kill "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$PROJECT_DIR/Backend"
"$VENV_DIR/bin/uvicorn" api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

for _ in {1..40}; do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "FastAPI failed to start. Check the backend error above."
    exit 1
  fi
  sleep 0.25
done

if ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "FastAPI did not become healthy on port 8000."
  exit 1
fi

cd "$FRONTEND_DIR"
npm run dev &
WEB_PID=$!

echo "API: http://127.0.0.1:8000"
echo "Web: http://localhost:3000"
wait

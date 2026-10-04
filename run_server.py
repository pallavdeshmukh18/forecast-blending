"""
SkyCast FastAPI Backend Server Runner.
Run this script from the project root to start the backend API server.
"""

import sys
from pathlib import Path
import uvicorn

_project_root = Path(__file__).resolve().parent
_backend_dir = _project_root / "Backend"
_ml_dir = _project_root / "ML"

for _p in [str(_ml_dir), str(_backend_dir), str(_project_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __name__ == "__main__":
    print("=" * 65)
    print("  Starting SkyCast Forecast Blending & Live Operational API")
    print("  Swagger UI Docs: http://127.0.0.1:8000/docs")
    print("  Live Status:    http://127.0.0.1:8000/api/live/status")
    print("=" * 65)
    uvicorn.run("Backend.api.main:app", host="127.0.0.1", port=8000, reload=True)

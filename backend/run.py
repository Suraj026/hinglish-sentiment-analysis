"""Entry-point: start the HinglishPulse FastAPI backend.

Usage (from repo root):
    python backend/run.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

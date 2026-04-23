import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


if __name__ == "__main__":
    # Get port from environment variable, default to 8000 for local development
    port = int(os.getenv("PORT", "8000"))
    # Disable reload in production (Railway will handle restarts)
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        app_dir="src",
    )


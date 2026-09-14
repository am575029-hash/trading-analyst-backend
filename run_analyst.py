from __future__ import annotations

import sys
import subprocess
import time
import io

# Fix Windows console encoding to handle UTF-8 symbols without UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "httpx",
    "pandas",
    "numpy"
]

def check_and_install_dependencies():
    """Verifies that all required packages for the 3-book AI engine are installed."""
    print("=" * 65)
    print("  AI TRADING ANALYST PRO - MULTI-BOOK HIGH-SPEED ENGINE")
    print("=" * 65)
    print("\n[1/3] Checking local Python runtime & dependencies...")

    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[*] Missing packages detected: {', '.join(missing)}")
        print("[*] Installing missing dependencies automatically...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("[OK] All packages installed successfully.")
        except Exception as err:
            print(f"[!] Failed to auto-install: {err}")
            print(f"[!] Please run: pip install {' '.join(missing)}")
            sys.exit(1)
    else:
        print("[OK] All mathematical and FastAPI dependencies verified.")

def start_server():
    """Launches the sub-20ms multi-engine FastAPI server."""
    print("\n[2/3] Initializing Trained Strategy Modules:")
    print("  |- [Book 1] Charles Kirkpatrick & Fidelity (Fidelity Chart Patterns)")
    print("  |- [Book 2] Roman Sadowski (9 Advanced Quantitative Systems)")
    print("  `- [Book 3] Mark Douglas (Trading in the Zone - Probabilistic Risk)")
    print("\n[3/3] Starting Local Server on http://127.0.0.1:8000 ...")
    print("=" * 65)
    print(">> Open your Floating App and click 'Analyze & Refresh Market'")
    print(">> Close the app window anytime to stop the background engine.")
    print("=" * 65 + "\n")

    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    check_and_install_dependencies()
    start_server()
import os
import time
from datetime import datetime

def log(msg: str):
    print(f"[{datetime.utcnow().isoformat()}Z] {msg}", flush=True)

if __name__ == "__main__":
    log("SCALP-AI PAPER MODE STARTED ✅")
    log(f"PYTHON={os.getenv('PYTHON_VERSION','unknown')}")

    while True:
        log("Bot çalışıyor... (paper) ✅")
        time.sleep(60)

"""Local configuration; never prints or persists secret values."""
import os
from pathlib import Path

DEFAULT_OPENAI_MODEL = "gpt-6.1-sol"

def load_local_env(path: Path) -> None:
    """Load simple KEY=VALUE pairs without adding another dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")

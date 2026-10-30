"""Version and build-commit discovery. The commit comes from pip's direct_url.json (a git install), else from the
source checkout's git HEAD, else 'unknown'. No network, no subprocess when installed from git."""
import importlib.metadata
import json
import subprocess
from pathlib import Path


def version() -> str:
    try:
        return importlib.metadata.version("liger-ce-fix")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def commit() -> str:
    try:
        text = importlib.metadata.distribution("liger-ce-fix").read_text("direct_url.json")
        if text:
            sha = json.loads(text).get("vcs_info", {}).get("commit_id")
            if sha:
                return sha
    except importlib.metadata.PackageNotFoundError:
        pass
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists():
        try:
            return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass
    return "unknown"

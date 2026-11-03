"""Small, shared host-side status persistence helpers."""

from datetime import datetime, timezone
import json
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def update_counts(run: dict) -> None:
    run["counts"] = {
        status: sum(trial["status"] == status for trial in run["trials"])
        for status in ("pending", "running", "completed", "incomplete", "cancelled")
    }

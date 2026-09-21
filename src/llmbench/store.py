"""Append-only experiment evidence with explicit attempt states and atomic artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict, is_dataclass

from .config import ExperimentManifest, canonical_json

STATES = {
    "planned": {"running", "cancelled"},
    "running": {"completed", "failed", "cancelled", "interrupted"},
    "completed": set(), "failed": set(), "cancelled": set(), "interrupted": set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root: str | Path, max_artifact_bytes: int = 1_073_741_824):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_artifact_bytes = max_artifact_bytes
        self._artifact_bytes = sum(path.stat().st_size for path in (self.root / "raw").rglob("*") if path.is_file())
        self.db = sqlite3.connect(self.root / "results.sqlite3", timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY, manifest TEXT NOT NULL, created TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS attempts (
            id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
            state TEXT NOT NULL, synthetic INTEGER NOT NULL, created TEXT NOT NULL,
            updated TEXT NOT NULL, parent_id TEXT REFERENCES attempts(id));
          CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id TEXT NOT NULL REFERENCES attempts(id),
            time TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS samples (
            attempt_id TEXT NOT NULL REFERENCES attempts(id), task_id TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(attempt_id, task_id));
          CREATE TABLE IF NOT EXISTS artifacts (
            attempt_id TEXT NOT NULL REFERENCES attempts(id), name TEXT NOT NULL,
            sha256 TEXT NOT NULL, size INTEGER NOT NULL, relative_path TEXT NOT NULL,
            PRIMARY KEY(attempt_id, name));
          CREATE TABLE IF NOT EXISTS campaigns (
            id TEXT PRIMARY KEY, elapsed_seconds REAL NOT NULL, candidates INTEGER NOT NULL,
            updated TEXT NOT NULL);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def create_attempt(self, manifest: ExperimentManifest, *, synthetic: bool, parent_id=None) -> str:
        if type(synthetic) is not bool:
            raise TypeError("synthetic must be a boolean")
        if manifest.backend.engine == "mock" and not synthetic:
            raise ValueError("mock runs must be synthetic")
        fingerprint = manifest.fingerprint()
        # Canonical immutable snapshot is authoritative, not a caller-owned object.
        payload = canonical_json(manifest.model_dump(mode="json"))
        attempt_id = uuid.uuid4().hex
        now = utc_now()
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO experiments VALUES (?,?,?)", (fingerprint, payload, now))
            if parent_id:
                parent = self.attempt(parent_id)
                if parent["state"] not in {"failed", "cancelled", "interrupted"}:
                    raise ValueError("only unsuccessful terminal attempts can be resumed")
                if parent["experiment_id"] != fingerprint:
                    raise ValueError("resume requires the identical manifest")
            self.db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?)",
                            (attempt_id, fingerprint, "planned", int(synthetic), now, now, parent_id))
            self._event(attempt_id, "created", {"parent_id": parent_id})
        return attempt_id

    def _event(self, attempt_id, kind, payload):
        self.db.execute("INSERT INTO events(attempt_id,time,kind,payload) VALUES (?,?,?,?)",
                        (attempt_id, utc_now(), kind, canonical_json(payload)))

    def event(self, attempt_id, kind, payload):
        with self.db:
            self._event(attempt_id, kind, payload)

    def attempt(self, attempt_id) -> dict:
        row = self.db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        return dict(row)

    def transition(self, attempt_id, target, reason=""):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            current = self.attempt(attempt_id)["state"]
            if target not in STATES[current]:
                raise ValueError(f"invalid transition {current} -> {target}")
            self.db.execute("UPDATE attempts SET state=?,updated=? WHERE id=?", (target, utc_now(), attempt_id))
            self._event(attempt_id, "state", {"from": current, "to": target, "reason": reason})

    def add_sample(self, attempt_id, sample: dict):
        if not sample.get("task_id"):
            raise ValueError("sample requires a stable task_id")
        with self.db:
            if self.attempt(attempt_id)["state"] != "running":
                raise ValueError("samples can only be appended to running attempts")
            self.db.execute("INSERT INTO samples VALUES (?,?,?)",
                            (attempt_id, sample["task_id"], canonical_json(sample)))

    def artifact(self, attempt_id: str, name: str, content: bytes) -> Path:
        self.attempt(attempt_id)
        if not name or Path(name).name != name or name in {".", ".."} or ":" in name or "\\" in name:
            raise ValueError("artifact name must be a single safe filename")
        # Count orphaned bytes too: a crash may occur after fsync but before indexing.
        used = sum(path.stat().st_size for path in (self.root / "raw").rglob("*") if path.is_file())
        self._artifact_bytes = max(self._artifact_bytes, used)
        if self._artifact_bytes + len(content) > self.max_artifact_bytes:
            raise ValueError("artifact budget exhausted")
        directory = self.root / "raw" / attempt_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / name
        digest = hashlib.sha256(content).hexdigest()
        # Exclusive creation preserves existing bytes even if the process died before DB insertion.
        with target.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        self._artifact_bytes += len(content)
        with self.db:
            self.db.execute("INSERT INTO artifacts VALUES (?,?,?,?,?)",
                            (attempt_id, name, digest, len(content), str(target.relative_to(self.root))))
        return target

    def results(self, attempt_id) -> dict:
        attempt = self.attempt(attempt_id)
        manifest = self.db.execute("SELECT manifest FROM experiments WHERE id=?",
                                   (attempt["experiment_id"],)).fetchone()[0]
        samples = [json.loads(row[0]) for row in self.db.execute(
            "SELECT payload FROM samples WHERE attempt_id=? ORDER BY task_id", (attempt_id,))]
        events = [dict(row) for row in self.db.execute(
            "SELECT * FROM events WHERE attempt_id=? ORDER BY seq", (attempt_id,))]
        return {"attempt": attempt, "manifest": json.loads(manifest), "samples": samples, "events": events}

    @contextmanager
    def trace(self, attempt_id, name="transport.jsonl"):
        """Stream raw evidence durably before parsing; seal even when evaluation raises."""
        if self.attempt(attempt_id)["state"] != "running":
            raise ValueError("trace requires a running attempt")
        if Path(name).name != name or ":" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("unsafe trace filename")
        used = sum(path.stat().st_size for path in (self.root / "raw").rglob("*") if path.is_file())
        self._artifact_bytes = max(self._artifact_bytes, used)
        directory = self.root / "raw" / attempt_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / name
        count = 0
        digest = hashlib.sha256()
        with target.open("xb") as handle:
            def write(event):
                nonlocal count
                payload = asdict(event) if is_dataclass(event) else event
                line = (canonical_json(payload) + "\n").encode("utf-8")
                if self._artifact_bytes + len(line) > self.max_artifact_bytes:
                    raise ValueError("raw trace artifact budget exhausted")
                handle.write(line)
                handle.flush()  # Preserve partial output through a process exception/crash.
                count += len(line)
                self._artifact_bytes += len(line)
                digest.update(line)
            try:
                yield write
            finally:
                handle.flush()
                os.fsync(handle.fileno())
                with self.db:
                    self.db.execute("INSERT INTO artifacts VALUES (?,?,?,?,?)",
                                    (attempt_id, name, digest.hexdigest(), count, str(target.relative_to(self.root))))

    def attempts(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM attempts ORDER BY created")]

    def attempts_in_insertion_order(self) -> list[dict]:
        """Attempts with their manifests in append order: timestamps can tie, the rowid cannot."""
        return [{**dict(row), "manifest": json.loads(row["manifest"])} for row in self.db.execute(
            "SELECT a.*, e.manifest FROM attempts a JOIN experiments e ON e.id=a.experiment_id ORDER BY a.rowid")]

    def campaign_state(self, campaign_id):
        row = self.db.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
        return dict(row) if row else {"elapsed_seconds": 0., "candidates": 0}

    def checkpoint_campaign(self, campaign_id, elapsed_seconds, candidates):
        with self.db:
            self.db.execute("INSERT INTO campaigns VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                            "elapsed_seconds=excluded.elapsed_seconds,candidates=excluded.candidates,updated=excluded.updated",
                            (campaign_id, elapsed_seconds, candidates, utc_now()))

    @contextmanager
    def campaign_lock(self, lock_path=None):
        """A stale lock needs explicit human/CLI recovery; never break an active lease automatically."""
        path = Path(lock_path) if lock_path is not None else self.root / "campaign.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            from .locks import describe_lock
            raise FileExistsError(f"another campaign holds the lock: {describe_lock(path)}") from exc
        try:
            os.write(descriptor, canonical_json({"pid": os.getpid(), "created": utc_now()}).encode())
            os.close(descriptor)
            descriptor = None
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            path.unlink(missing_ok=True)

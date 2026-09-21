"""Capture exact Python dependencies and harness source without contacting any service."""

import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path

from .config import canonical_json


def environment_record() -> dict:
    packages = sorted({(dist.metadata["Name"].lower().replace("_", "-").replace(".", "-"), dist.version)
                       for dist in importlib.metadata.distributions() if dist.metadata["Name"]})
    source = Path(__file__).parent
    files = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(source.rglob("*.py"))}
    record = {"python": platform.python_version(), "implementation": sys.implementation.name,
              "platform": platform.platform(), "packages": dict(packages), "source_files": files}
    return {**record, "sha256": hashlib.sha256(canonical_json(record).encode()).hexdigest()}


def assert_comparable(left: dict, right: dict, family: str):
    """Reject comparisons when anything outside the declared treatment differs."""
    import copy
    from .search import propose
    from .config import ExperimentManifest

    # Reuse proposal allowlists by validating every differing top-level/backend field.
    a, b = copy.deepcopy(left), copy.deepcopy(right)
    for item in (a, b):
        item.pop("annotations", None)
        if item.get("environment_hash") in (None, "unrecorded"):
            raise ValueError("comparison environment must be recorded")
    differences = {}
    for key in set(a) | set(b):
        if a.get(key) == b.get(key):
            continue
        if key == "backend":
            for setting in set(a[key]) | set(b[key]):
                if a[key].get(setting) != b[key].get(setting):
                    differences[f"backend.{setting}"] = b[key].get(setting)
        else:
            differences[key] = b.get(key)
    if not differences:
        return
    proposed = propose(ExperimentManifest.model_validate_json(canonical_json(left)), differences, family)
    if proposed.model_dump(mode="json", exclude={"annotations"}) != b:
        raise ValueError("comparison differs outside treatment fields")

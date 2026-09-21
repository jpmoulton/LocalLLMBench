import json

import pytest

from llmbench.coding.image_plan import verified_local_image_id, worker_image_build_plan
from llmbench.coding.sandbox import BoundedProcessExecutor, DockerWorker
from llmbench.safety import SessionLock


BASE = "node:24-bookworm-slim@sha256:" + "a" * 64


def test_build_plan_is_pure_and_retains_exact_inputs(tmp_path):
    target = tmp_path / "not-created"
    plan = worker_image_build_plan(base_image=BASE, artifact_dir=target)
    assert not target.exists()
    assert plan["status"] == "planned-not-executed"
    assert plan["base_image"] == BASE and plan["final_image_id"] is None
    assert plan["build_argv"][0:2] == ["docker", "build"]
    assert "BASE_IMAGE=" + BASE in plan["build_argv"]
    assert plan["typescript_version"] == "5.9.3"
    assert plan["build_requires_network"] is True
    assert plan["runtime"]["network"] == "none"
    assert plan["runtime"]["memory_mib"] == 512 and plan["runtime"]["gpu_devices"] == []
    # every file the build context admits is hashed, so a changed toolchain pin is a changed plan
    assert set(plan["source_sha256"]) == {"Dockerfile", ".dockerignore", "polyglot-js/package.json"}
    json.dumps(plan)


@pytest.mark.parametrize("base", ["node:24-bookworm-slim", "node:latest@sha256:" + "a" * 64,
                                  "evil.example/node:24-bookworm-slim@sha256:" + "a" * 64,
                                  "node:24-bookworm-slim@sha256:" + "a" * 63])
def test_build_plan_rejects_mutable_or_wrong_base(tmp_path, base):
    with pytest.raises(ValueError, match="official"):
        worker_image_build_plan(base_image=base, artifact_dir=tmp_path)


def test_local_image_id_can_plan_worker_without_registry_publication(tmp_path):
    iid = "sha256:" + "b" * 64
    path = tmp_path / "worker-image.id"
    path.write_text(iid + "\n", encoding="ascii")
    assert verified_local_image_id(path) == iid
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    worker = DockerWorker(allowed_root=tmp_path, session_lock=SessionLock())
    job = worker.prepare(candidate, image=iid, command=("python3", "/workspace/.llmbench-driver.py"),
                         collect_result=True)
    BoundedProcessExecutor._validate(job.argv)
    assert iid in job.argv and "--pull=never" in job.argv


@pytest.mark.parametrize("data", [b"llmbench-worker:staged", b"sha256:" + b"1" * 64 + b"\nother",
                                  b"sha256:" + b"1" * 65, b"x" * 81, b"\xff"])
def test_image_id_rejects_tags_multiple_records_and_oversize(tmp_path, data):
    path = tmp_path / "worker-image.id"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        verified_local_image_id(path)

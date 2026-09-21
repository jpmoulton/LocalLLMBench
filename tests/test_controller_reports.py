import json

import pytest

from llmbench.cli import main
from llmbench.config import AcceptancePolicy, CampaignPolicy, RunMode
from llmbench.controller import execute_attempt, generate_candidates, run_campaign
from llmbench.provenance import assert_comparable, environment_record
from llmbench.reports import export_preset, html_report, write_reports
from llmbench.safety import OperationForbidden
from llmbench.store import Store


def test_environment_is_stable_and_exact():
    first = environment_record()
    assert first == environment_record()
    # Exact versions are recorded, whatever they are on this machine; pydantic is a hard dependency, so it is
    # always there to check. Pinning one particular release here only ever tested the author's virtualenv.
    assert first["packages"]["pydantic"].split(".")[0] == "2"
    assert len(first["sha256"]) == 64


def test_compare_rejects_environment_or_prompt_changes(manifest):
    a = json.loads(manifest.model_dump_json())
    a["environment_hash"] = "recorded"
    b = json.loads(json.dumps(a))
    b["backend"]["k_cache"] = "q8_0"
    assert_comparable(a, b, "kv")
    b["generation"]["seed"] = 4
    with pytest.raises(ValueError):
        assert_comparable(a, b, "kv")


def test_live_denied_before_executor(manifest, tmp_path):
    data = json.loads(manifest.model_dump_json())
    data["backend"]["engine"] = "lm-studio"
    data["mode"] = "live"
    from llmbench.config import ExperimentManifest
    live = ExperimentManifest.model_validate_json(json.dumps(data))
    calls = []
    with Store(tmp_path) as store:
        with pytest.raises(OperationForbidden):
            execute_attempt(store, live, CampaignPolicy(name="blocked", mode=RunMode.LIVE),
                            lambda *a, **kw: calls.append(1))
        assert not store.attempts()
    assert calls == []


def test_cli_doctor_does_not_probe(monkeypatch, capsys):
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("network prohibited"))
    assert main(["doctor"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["server_contacted"] is False
    from llmbench.safety import SessionLock
    assert result["session_policy"] == vars(SessionLock.read("runtime-policy.json"))


def test_campaign_demo_and_resume(manifest, tmp_path):
    policy = CampaignPolicy(name="test")
    manifests = generate_candidates(manifest, [{"family": "kv", "changes": {"backend.k_cache": "q8_0"}}])
    with Store(tmp_path) as store:
        campaign = run_campaign(store, manifests, policy)
        assert len(campaign["results"]) == 2
        assert all(row["status"] == "completed" for row in campaign["results"])
        assert not campaign["frontier"]
        assert all(not row["eligibility"]["eligible"] for row in campaign["results"])
        again = run_campaign(store, manifests, policy, resume=True)
        assert [row["attempt_id"] for row in again["results"]] == [row["attempt_id"] for row in campaign["results"]]
        stricter = CampaignPolicy(name="test", acceptance=AcceptancePolicy(minimum_tokens_per_second=120.))
        changed = run_campaign(store, manifests, stricter, resume=True)
        assert changed["results"][0]["attempt_id"] != campaign["results"][0]["attempt_id"]
        files = write_reports(campaign, tmp_path / "reports")
        assert "SYNTHETIC" in __import__("pathlib").Path(files["markdown"]).read_text()
        evidence = campaign["results"][0]
        with pytest.raises(ValueError, match="authoritative"):  # Legacy call without Store context.
            export_preset(manifest.model_dump(mode="json"), evidence, policy, tmp_path / "preset.json")
        # With full context the synthetic evidence itself is judged, and refused for its origin.
        with pytest.raises(ValueError, match="supplied:not_completed_measured_evidence"):
            export_preset(store.results(evidence["attempt_id"])["manifest"], evidence, policy,
                          tmp_path / "preset.json", store=store, campaign=campaign,
                          baseline_attempt_id=campaign["results"][0]["attempt_id"], allow_self_reference=True)
        assert not (tmp_path / "preset.json").exists()


def test_global_lock_shared_between_stores(tmp_path):
    with Store(tmp_path / "one") as one, Store(tmp_path / "two") as two:
        shared = tmp_path / "gpu.lock"
        with one.campaign_lock(shared):
            with pytest.raises(FileExistsError):
                with two.campaign_lock(shared):
                    pass


def test_report_html_escapes_untrusted_data():
    report = html_report({"campaign_id": "<script>alert(1)</script>", "results": []})
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;" in report


def test_interrupt_records_elapsed_and_crash_reservation(manifest, tmp_path, monkeypatch):
    import llmbench.controller as controller
    from llmbench.search import BudgetTracker
    now = [0.0]
    observed_reservations = []
    monkeypatch.setattr(controller, "BudgetTracker", lambda limits: BudgetTracker(limits, clock=lambda: now[0]))
    def interrupted(store, *args, **kwargs):
        observed_reservations.append(store.db.execute("SELECT elapsed_seconds FROM campaigns").fetchone()[0])
        now[0] += 60
        raise KeyboardInterrupt()
    monkeypatch.setattr(controller, "execute_attempt", interrupted)
    with Store(tmp_path) as store:
        with pytest.raises(KeyboardInterrupt):
            controller.run_campaign(store, [manifest], CampaignPolicy(name="interrupt"))
        assert observed_reservations == [120]
        assert store.db.execute("SELECT elapsed_seconds FROM campaigns").fetchone()[0] == 60
        assert not (tmp_path / "campaign.lock").exists()

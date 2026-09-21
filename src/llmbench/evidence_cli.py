"""Read-only commands over recorded evidence: nothing here contacts a server, a GPU or Docker.

``analyze`` is the useful one: it re-scores a finished campaign from its result store under a different
policy, so acceptance thresholds can be recalibrated without spending another GPU hour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COMMANDS = ("doctor", "list", "show", "analyze", "report")


def add_evidence_commands(sub: argparse._SubParsersAction) -> None:
    doctor = sub.add_parser("doctor", help="Show package versions and the runtime policy; contacts nothing")
    doctor.add_argument("--policy", default="runtime-policy.json")
    for name, text in (("list", "List the attempts recorded in a result store"),
                       ("show", "Show one recorded attempt with its samples and events")):
        command = sub.add_parser(name, help=text)
        command.add_argument("--store", required=True, help="a run directory holding results.sqlite3")
        if name == "show":
            command.add_argument("attempt_id")
    analyze = sub.add_parser("analyze", help="Re-score a finished campaign under a (different) policy; no GPU")
    analyze.add_argument("campaign", help="reports/campaign.json from a finished run")
    analyze.add_argument("--store", required=True, help="the run directory holding results.sqlite3")
    analyze.add_argument("--policy", required=True, help="a CampaignPolicy JSON file")
    analyze.add_argument("--baseline", help="attempt id to compare against; default: the campaign's own")
    analyze.add_argument("--output", required=True)
    report = sub.add_parser("report", help="Rebuild the JSON/Markdown/HTML report views from a campaign JSON")
    report.add_argument("campaign")
    report.add_argument("--output", required=True)


def run_evidence_command(args: argparse.Namespace) -> Any:
    from .reports import write_reports
    from .store import Store
    if args.command == "doctor":
        from .provenance import environment_record
        from .safety import SessionLock
        return {"environment": environment_record(), "session_policy": vars(SessionLock.read(args.policy)),
                "server_contacted": False, "model_operations_performed": False}
    if args.command in ("list", "show"):
        if not (Path(args.store) / "results.sqlite3").exists():
            raise ValueError(f"no result store at {args.store}")
        with Store(args.store) as store:
            return store.attempts() if args.command == "list" else store.results(args.attempt_id)
    campaign = json.loads(Path(args.campaign).read_text(encoding="utf-8"))
    if args.command == "analyze":
        from .analysis import analyze_campaign
        from .config import CampaignPolicy
        policy = CampaignPolicy.model_validate_json(Path(args.policy).read_text(encoding="utf-8"))
        with Store(args.store) as store:
            campaign = analyze_campaign(store, campaign, policy, args.baseline)
    return write_reports(campaign, args.output)

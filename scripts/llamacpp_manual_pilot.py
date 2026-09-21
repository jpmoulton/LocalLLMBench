"""Root-owned manual llama.cpp container pilot: readback, native speed probes and one native tool call.

Diagnostic only. It talks to an already running, root-owned llama-server and saves raw responses with
exclusive filenames. It is not the container runner and produces no validated evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

PROMPT = ("Write a detailed technical explanation of how a B-tree index works in a relational database, "
          "including node splits, merges, search, range scans and why it suits disk storage.")
TOOLS = [{"type": "function", "function": {
    "name": "create_ticket", "description": "Create a support ticket.",
    "parameters": {"type": "object", "additionalProperties": False,
                   "required": ["title", "priority", "tags", "assignee"],
                   "properties": {"title": {"type": "string"},
                                  "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                                  "tags": {"type": "array", "items": {"type": "string"}},
                                  "assignee": {"type": "object", "additionalProperties": False,
                                               "required": ["team", "id"],
                                               "properties": {"team": {"type": "string"},
                                                              "id": {"type": "integer"}}}}}}}]


def save(directory: Path, name: str, value) -> None:
    with (directory / name).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=512)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=args.url, timeout=httpx.Timeout(600, connect=10))
    for name in ("props", "slots", "v1/models", "health"):
        response = client.get("/" + name)
        save(out, name.replace("/", "-") + ".json", {"status": response.status_code, "body": response.json()})
    summary = {"speed": [], "tool": None}
    for index in range(args.repetitions):
        started = time.perf_counter()
        first = None
        chunks = []
        payload = {"prompt": PROMPT, "n_predict": args.tokens, "ignore_eos": True, "temperature": 0,
                   "seed": 42, "cache_prompt": False, "stream": True, "timings_per_token": False}
        with client.stream("POST", "/completion", json=payload) as response:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                now = time.perf_counter()
                event = json.loads(line[6:])
                if first is None and event.get("content"):
                    first = now - started
                chunks.append({"t": now - started, "event": event})
        wall = time.perf_counter() - started
        final = chunks[-1]["event"]
        save(out, f"speed-{index}.json", {"request": payload, "wall_seconds": wall, "first_content_seconds": first,
                                          "events": chunks})
        timings = final.get("timings", {})
        summary["speed"].append({"index": index, "wall_seconds": wall, "first_content_seconds": first,
                                 "tokens_predicted": final.get("tokens_predicted"),
                                 "tokens_evaluated": final.get("tokens_evaluated"), "truncated": final.get("truncated"),
                                 "stop_type": final.get("stop_type"), "timings": timings})
        print(f"speed[{index}] native predicted_per_second={timings.get('predicted_per_second')} "
              f"prompt_per_second={timings.get('prompt_per_second')} n={timings.get('predicted_n')} wall={wall:.2f}s")
    request = {"model": "pilot", "temperature": 0, "seed": 42, "max_tokens": 1024, "tools": TOOLS,
               "messages": [{"role": "user", "content":
                             "File a high priority ticket titled 'Checkout 500 on submit' tagged payments and "
                             "regression, assigned to team 'web-platform' member id 4172. Use the tool."}]}
    started = time.perf_counter()
    response = client.post("/v1/chat/completions", json=request)
    body = response.json()
    save(out, "tool-call.json", {"request": request, "status": response.status_code,
                                 "wall_seconds": time.perf_counter() - started, "body": body})
    calls = (body.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
    parsed = [json.loads(call["function"]["arguments"]) for call in calls]
    expected = {"title": "Checkout 500 on submit", "priority": "high", "tags": ["payments", "regression"],
                "assignee": {"team": "web-platform", "id": 4172}}
    summary["tool"] = {"calls": len(calls), "arguments": parsed, "exact_match": parsed == [expected],
                       "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
                       "usage": body.get("usage"), "timings": body.get("timings")}
    print("tool exact_match=", summary["tool"]["exact_match"], "calls=", len(calls))
    save(out, "summary.json", summary)


if __name__ == "__main__":
    main()

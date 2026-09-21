"""Root-owned diagnostic: native decode/prefill rate with an actually filled prompt on a running llama-server.

Builds a prompt of an exact token count through the server's own /tokenize + /detokenize endpoints, then
runs /completion with caching disabled. Diagnostic pilot only; saves raw output with exclusive filenames.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

PARAGRAPH = ("The scheduler assigns each job a lease, records its heartbeat, and requeues work whose lease expired. "
             "Workers acknowledge results idempotently so that a retried delivery never applies an update twice. ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=args.url, timeout=httpx.Timeout(1800, connect=10))
    per = len(client.post("/tokenize", json={"content": PARAGRAPH}).json()["tokens"])
    text = PARAGRAPH * (args.input_tokens // per + 2)
    tokens = client.post("/tokenize", json={"content": text, "add_special": True}).json()["tokens"][: args.input_tokens]
    results = []
    for index in range(args.repetitions):
        payload = {"prompt": tokens, "n_predict": args.tokens, "ignore_eos": True, "temperature": 0, "seed": 42,
                   "cache_prompt": False}
        started = time.perf_counter()
        body = client.post("/completion", json=payload).json()
        wall = time.perf_counter() - started
        record = {"index": index, "requested_input_tokens": len(tokens), "wall_seconds": wall,
                  "tokens_evaluated": body.get("tokens_evaluated"), "tokens_predicted": body.get("tokens_predicted"),
                  "truncated": body.get("truncated"), "stop_type": body.get("stop_type"),
                  "timings": body.get("timings")}
        results.append(record)
        with (out / f"filled-{len(tokens)}-{index}.json").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"record": record, "body": {k: v for k, v in body.items() if k != "prompt"}},
                                    indent=2, ensure_ascii=False))
        timings = body.get("timings", {})
        print(f"in={body.get('tokens_evaluated')} out={body.get('tokens_predicted')} truncated={body.get('truncated')} "
              f"decode={timings.get('predicted_per_second'):.2f} tok/s prefill={timings.get('prompt_per_second'):.1f} "
              f"tok/s wall={wall:.1f}s")
    with (out / f"filled-{len(tokens)}-summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

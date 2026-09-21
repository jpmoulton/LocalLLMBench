"""Root-owned S1-00 live route checks against a running llama-server (diagnostic; exclusive output files).

(b) /apply-template + /tokenize count versus chat usage.prompt_tokens, with and without tools.
(c) streamed chat: does the final chunk carry timings; are ignore_eos and cache_prompt honoured.
(d) deliberate overflow: status code and error body.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a record by id.",
          "parameters": {"type": "object", "required": ["id"], "additionalProperties": False,
                         "properties": {"id": {"type": "integer"}}}}}]
MESSAGES = [{"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Reply with the single word: ready."}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=args.url, timeout=httpx.Timeout(300, connect=10))
    report: dict = {}

    count = {}
    for label, tools in (("no_tools", None), ("tools", TOOLS)):
        body = {"messages": MESSAGES} | ({"tools": tools} if tools else {})
        rendered = client.post("/apply-template", json=body)
        template = rendered.json()
        prompt = template.get("prompt") if isinstance(template, dict) else None
        variants = {}
        for add_special in (True, False):
            tokens = client.post("/tokenize", json={"content": prompt, "add_special": add_special,
                                                    "parse_special": True}).json()["tokens"]
            variants[f"add_special={add_special}"] = len(tokens)
        chat = client.post("/v1/chat/completions", json=body | {"max_tokens": 1, "temperature": 0,
                                                                "cache_prompt": False}).json()
        count[label] = {"apply_template_status": rendered.status_code,
                        "apply_template_keys": sorted(template) if isinstance(template, dict) else None,
                        "rendered_prompt": prompt, "tokenize_counts": variants,
                        "usage": chat.get("usage"), "timings": chat.get("timings")}
    report["count_route"] = count

    payload = {"messages": [{"role": "user", "content": "Count from 1 to 5."}], "max_tokens": 96, "temperature": 0,
               "seed": 42, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
               "cache_prompt": False}
    chunks = []
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        status = response.status_code
        for line in response.iter_lines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                chunks.append(json.loads(line[6:]))
    with_timings = [index for index, chunk in enumerate(chunks) if "timings" in chunk]
    with_usage = [index for index, chunk in enumerate(chunks) if chunk.get("usage")]
    finish = [(index, chunk["choices"][0].get("finish_reason")) for index, chunk in enumerate(chunks)
              if chunk.get("choices") and chunk["choices"][0].get("finish_reason")]
    delta_keys = sorted({key for chunk in chunks for choice in chunk.get("choices", [])
                         for key in choice.get("delta", {})})
    report["stream_chat"] = {"status": status, "chunks": len(chunks), "chunks_with_timings": with_timings[-3:],
                             "chunks_with_usage": with_usage, "finish": finish, "delta_keys": delta_keys,
                             "last_chunk": chunks[-1], "second_last_chunk": chunks[-2] if len(chunks) > 1 else None,
                             "first_chunk": chunks[0]}
    second = client.post("/v1/chat/completions", json=payload | {"stream": False}).json()
    report["repeat_nonstream_cache"] = {"usage": second.get("usage"), "timings": second.get("timings"),
                                        "finish_reason": second["choices"][0].get("finish_reason")}

    n_ctx = client.get("/props").json()["default_generation_settings"]["n_ctx"]
    overflow = client.post("/completion", json={"prompt": [1000] * (n_ctx + 64), "n_predict": 1,
                                                "cache_prompt": False})
    report["overflow_completion"] = {"n_ctx": n_ctx, "status": overflow.status_code, "body": overflow.text[:2000]}
    big = "word " * (n_ctx + 64)
    overflow_chat = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": big}],
                                                              "max_tokens": 1, "cache_prompt": False})
    report["overflow_chat"] = {"status": overflow_chat.status_code, "body": overflow_chat.text[:2000]}

    with (out / "route-checks.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False))
    summary = {"count": {k: (v["tokenize_counts"], (v["usage"] or {}).get("prompt_tokens")) for k, v in count.items()},
               "stream": {k: report["stream_chat"][k] for k in ("chunks", "chunks_with_timings", "chunks_with_usage",
                                                                "finish", "delta_keys")},
               "stream_last_timings": chunks[-1].get("timings"), "stream_last_usage": chunks[-1].get("usage"),
               "repeat": report["repeat_nonstream_cache"],
               "overflow_completion": report["overflow_completion"], "overflow_chat": report["overflow_chat"]}
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()

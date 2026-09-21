# Third-party notices

**Vendored in this repository**

| What | Where | Source and license |
|---|---|---|
| Word lists used by RULER's generators | `src/llmbench/benchmarks/ruler_tasks/wonderwords_lists.py` | `wonderwords` 3.0.1 by mrmaxguns — MIT. Copied byte-for-byte; each list carries its upstream SHA-256. |
| RULER task generators, templates and scorers (reimplemented to match upstream) and pinned reference text | `src/llmbench/benchmarks/ruler_tasks/`, `tests/data/ruler-upstream-*` | NVIDIA/RULER — Apache-2.0 |
| llama.cpp server help text and API response captures used as test fixtures | `tests/data/` | ggml-org/llama.cpp — MIT |

**Downloaded by `scripts/stage_benchmark_datasets.py`, never redistributed here**

| Dataset | Source | License |
|---|---|---|
| Berkeley Function Calling Leaderboard | ShishirPatil/gorilla | Apache-2.0 |
| MBPP+ | evalplus/mbppplus_release | Apache-2.0 (MBPP: CC-BY-4.0) |
| Aider Polyglot exercises (Exercism) | Aider-AI/polyglot-benchmark | MIT (Exercism content) |
| Essays used as RULER haystack text | gkamradt/needle-in-a-haystack | see upstream |

Check each upstream's current terms before redistributing staged data or publishing scores as official results.
Scores produced here use subsets and documented deviations (`docs/benchmarks.md`) and are not leaderboard scores.

# Benchmarks

Four public suites plus a few small local fixtures. No LLM judge is used anywhere; every score is mechanical.
Datasets are fetched once by `scripts/stage_benchmark_datasets.py`, pinned by per-file SHA-256 in a manifest, and
never fetched at run time (`--offline-verify` re-checks a staging directory without the network).

| Suite | Measures | Pinned source | Items |
|---|---|---|---|
| **BFCL** v4 (AST subset) | native tool calling: right function, right typed arguments, and *not* calling when nothing fits | `ShishirPatil/gorilla` | 3 per category by default (~21); `items_per_category` raises it |
| **RULER** | long-context retrieval and tracking, at each length up to the candidate's context | generators vendored from `NVIDIA/RULER` @ `c3f5e3b4…`; haystack essays from `gkamradt/needle-in-a-haystack` | tasks × lengths |
| **EvalPlus MBPP+** v0.2.0 | Python function synthesis against expanded tests (pass@1) | `evalplus/mbppplus_release` | 20 by default of 378; `item_limit` raises it |
| **Aider Polyglot** | editing real exercises with search/replace blocks, with one retry after seeing test output | `Aider-AI/polyglot-benchmark` @ `7e0611e7…` | Python + JavaScript: 83 exercises, split into development and holdout halves |

Local fixtures (`tool-probes`, `tool-episodes`, `niah`, `coding`) are hand-written smoke tests. Strong models score
1.000 on all of them, so they catch breakage but rank nothing; that is why the public suites exist.

Every adapter must return one row per declared task, failures included, so a crash can never shrink a
denominator; a task that did not really run is `environment_error`, never a silent zero presented as a score.
EvalPlus, Aider Polyglot and the `coding` fixtures execute generated code, which only ever happens in the sandboxed
worker container. When that sandbox is unavailable (a Mac without Docker, for example) they are never run on the
host instead. A plan that already knows it lists them as `blocked` with the reason and does not select them; a
candidate whose sandbox probe fails at run time records their rows as `environment_error` with a
`sandbox_unavailable: <reason>` reason. Those rows never enter a score: the candidate result, the candidate,
session, campaign, cross-model and coding reports and the sampling report leave them out of the category and print
it as `blocked` (unmeasured), never as a coding score of zero. One known gap: the campaign analysis behind a
session's eligibility (`controller`/`analysis`) still counts such rows as failures, so a candidate whose sandbox
disappeared mid-session cannot meet a coding floor there. Derived sessions never select blocked suites, so this
only arises when Docker goes away during a run.

## Where this differs from upstream, on purpose

**RULER.** Templates, answer prefixes, the depth ladder, word pools and scorers were checked against upstream's
sources and against OpenCompass's independent port; `tests/data/ruler-upstream-*` pins that reference text and the
tests execute upstream's own scoring functions beside ours. Two deviations: output caps are raised
(`ruler_output_tokens`) because upstream's 128/30/120-token caps were measured truncating *correct* answers into
failures, and a labelled lenient score (one Markdown fence stripped) is reported beside the strict one. Only
`niah_multikey_3` and `vt` run by default; `cwe` is output-cap sensitive and a weak discriminator. **Compare a
RULER score only with another score at the same length and task** — the per-length curve is the result, not an
average.

**BFCL.** The AST matcher and typed argument comparison are reimplemented for the single-turn categories. Native
mode (real tool calls) is the default; prompt mode is available but its numbers are this project's, not
leaderboard scores.

**EvalPlus.** Tests run in the sandboxed worker, one container per item, with a per-test timeout. The test module
is rebuilt on the host from the pinned dataset; the evaluator sends only `solution.py`.

**Aider Polyglot.** Only Python and JavaScript have pinned test commands (the other four upstream languages are
refused rather than guessed). Exercism ships every JavaScript test after the first as skipped (`xtest`); like
upstream's harness we un-skip them, otherwise an exercise "passes" on its first test alone. Unlike upstream, exit
code 0 is not enough to pass: the runner's summary must also report passing tests. `javascript/ledger` is a
refactoring exercise whose starting code already passes — a free point by design.

## Calibrating the harness, not just the model

`scripts/polyglot_reference_calibration.py` runs every exercise's *reference* solution and its untouched stub
through the real worker. All references must pass and all stubs must fail. It exists because a sandbox limit once
failed a correct solution (see [lessons](lessons.md)), and nothing in a model run can reveal that kind of bug.

## Statistical power

Defaults are sized to fit a time budget, not to separate close candidates. With *n* items the standard error is
roughly `sqrt(p(1-p)/n)`: about ±8 points at n=24, ±6 at n=60, ±3 at n=145. Differences inside that are noise.
Compare candidates on the *same items* with a paired test (`scripts/coding_report.py`), and never quote the pass
rate "among the items that finished" — the unfinished ones are the hard ones.

## Holdout

Suites marked holdout-capable partition their items deterministically by seed into development and holdout
sets. A holdout pass shows a result was not tuned to particular items. It does **not** show the items are unseen
by the model: public benchmark data may be in its training set.

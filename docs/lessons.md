# Lessons

Every entry is a mistake this project actually made while measuring a real model, and the guard that now exists
because of it. Comments in the source cite these as `LIVE-nnn`. Almost all of them were invisible to unit tests
and appeared only on the first real run — which is the first lesson.

## Traps that manufacture fake findings

**An output cap is not a wrong answer.** Three separate times a token cap turned correct behaviour into a
"quality result": RULER's `cwe` at 120 tokens, RULER's `vt` at 30, and reasoning mode on a coding suite at 1024,
where 9 of 20 responses were cut off mid-thought and reasoning *appeared* to hurt. *Guard:* `finish_reason` is
recorded for every response and reports print the cut-off count beside every score. Check it before believing any
score below 1.0.

**Do not quote the pass rate "among the items that finished".** After raising that cap, 10 of 11 finished items
passed and reasoning looked much better than baseline. On 60 identical items it was 42 vs 43: the items a
configuration fails to finish are the hard ones. *Guard:* compare on identical item sets with a paired test
(`scripts/coding_report.py`).

**A crash writes zeros.** A candidate that died mid-stage still recorded `score: 0.0` for every category it had
started. *Guard:* reports suppress scores for failed candidates and print `unmeasured`.

**"Not measured" is not "failed".** With no coding benchmark configured, every candidate was marked as failing the
coding threshold, so no candidate survived screening, no holdout ever ran, and `winner` was always `none`. *Guard:* a
category with no declared task is `unmeasured`: it cannot fail a candidate, it is listed in the eligibility record
and in every preset, and a validated-preset export still refuses a candidate that has one.

**A gate can be unreachable by arithmetic.** Non-inferiority is tested exactly: with *no* regression on *n* pass/fail
items, the smallest loss that can be excluded is `1 - 0.025^(1/n)` - 14% at 24 items, 2% only at 183. The policy
asks for 2% and the default plan picks about 24, so only the baseline could ever be validated, and nothing said so.
*Guard:* the plan warns before the run, every comparison reports `smallest_demonstrable_loss` and
`tasks_needed_for_maximum_loss`, and the recommendation names a faster or larger candidate that was measured but
could not be validated instead of showing only the baseline. The test itself was not loosened.

**Thresholds tuned on toy fixtures reject real models.** Floors of 0.95 made sense for three hand-written probes a
good model aces; a strong 27B model scores about 0.79 on BFCL. *Guard:* the floors are sanity floors derived from
measured baselines (`config.AcceptancePolicy`), and the paired comparison does the ranking.

**Most of a test file may be switched off.** 832 of 881 JavaScript tests in the Exercism corpus ship as `xtest`.
Run as shipped, an exercise passes on one test. *Guard:* they are un-skipped on the host's copy, as upstream does.

**Exit code 0 is not a pass.** `os._exit(0)` at import returns 0 with no tests run. *Guard:* a pass also needs the
runner's own summary line.

**Your sandbox can fail a correct answer.** A fixed open-files limit killed the *reference* solution of an exercise
that starts 50 worker threads. *Guard:* `scripts/polyglot_reference_calibration.py` runs every reference solution
through the real sandbox; the limit is now a bounded setting.

**Small samples lie politely.** On 8 retrieval items the *same model* scored 0.625, 0.750 and 1.000 across its own
candidates. On 24 tool items, 28 candidates across 4 models all landed within two items of each other. Neither
can rank anything.

## Traps in the harness

| id | What happened | Guard |
|---|---|---|
| LIVE-004 | Correct answers wrapped in a Markdown fence failed strict string matching | a labelled lenient score is reported beside the strict one, never instead of it |
| LIVE-005 | The wheel build needed `setuptools` the virtualenv did not have | `prepare --wheel` accepts a prebuilt wheel |
| LIVE-006 | `docker compose` rejected a pids limit declared twice | declared once, under `deploy.resources.limits` |
| LIVE-007 | A later stage reused an earlier stage's *closed* capture sink | every stage and benchmark opens its own trace sink |
| LIVE-008 | The tool schema serialised with different key order on the counting and serving paths: a one-token disagreement | the served payload is rebuilt through the same models; a small bounded slack is tolerated and never licenses a context claim |
| LIVE-009 | A shell treated Docker's warnings on stderr as fatal and skipped cleanup, leaving a container holding 28 GB of VRAM | drivers are plain Python; interrupts are forwarded to the session and waited for |
| LIVE-010 | With partial GPU offload the KV cache is split across devices; a two-state "on GPU?" flag misreported it | placement is `gpu`, `ram` or `split`, read from the server log |
| LIVE-011 | Holdout validation demanded categories the holdout could not contain, so it could never pass | validation is scoped to the categories the holdout covers, and says which |
| LIVE-012 | Editing source during a session changed `environment_hash` and broke `resume` | by design: do not edit the package while a session runs |
| LIVE-013 | Benchmark options were dropped when a selection round-tripped through the campaign manifest, silently reverting to defaults | options are part of the fingerprint and the run aborts if they are not carried |
| LIVE-014 | A coding benchmark adapter and the sandbox broker had been built against different contracts; every item errored and a fake score of 0.13 was reported | one bridge (`coding/benchmark_items.py`) with tests, and `environment_error` rows are never scored |
| LIVE-015 | A 262144-token candidate ran ten requests filled to that length (four minutes each) inside a fixed 20-minute bound; the stage died on every model and the context was never verified | `proposals.size_long_context_work`: fewer speed repetitions, the NIAH variants that fit, and bounds sized to the prompt |
| LIVE-016 | A process killed mid-run left a GPU lock that surfaced as a bare `FileExistsError`; the blocked launch still wrote a ledger, so the retry was told to `resume` a session that never started | the error names the holder and whether it is still running (`locks.py`); a new session checks the lock before writing anything |
| LIVE-017 | Completed holdout attempts ran again on every `resume` - unreachable until a candidate could survive screening | a completed holdout is reused exactly like a completed development attempt |

Also learned the hard way: stopping a sweep by killing its wrapper leaves the GPU container running. Stop the
session itself, and check `docker ps` afterwards.

## Things that are easy to get wrong about the hardware

* A quantization named in a filename may not be what the header declares. NVFP4 GGUFs store weights in a tensor
  type the `general.file_type` enum has no entry for, so the header says `Q8_0`. The quantization is therefore read
  from the tensors: the session records `NVFP4`, keeps the header's word as `declared_quantization`, and stores the
  tensor-type counts. A type this version does not know is reported by number, never guessed.
* Single-axis sweeps inherit the baseline. A 262144-token candidate at `f16` KV will not fit a 32 GB card beside a
  27B model; with a `q4_0` baseline it does.
* Different fine-tunes of one model can ship different chat templates. Tool-calling scores then differ for reasons
  that have nothing to do with the weights.
* A measurement taken while something else uses the GPU is not a measurement. Admission refuses to start when
  foreign VRAM exceeds `limits.max_foreign_vram_mib`.

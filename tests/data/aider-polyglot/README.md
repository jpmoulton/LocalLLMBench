# Hand-made Aider Polyglot corpus fixture

Every file under `tests/data/aider-polyglot/` was written by hand for these tests. Nothing here is copied
from `github.com/Aider-AI/polyglot-benchmark` or from Exercism: the exercises, instructions, stubs and
unit tests are invented, and the slugs deliberately do not match upstream exercise names. Only the
*layout* mirrors upstream (`<language>/exercises/practice/<slug>/` with `.docs/instructions.md`,
`.meta/config.json`, stub files and test files) so the loader is exercised against the real shape.

`pin.json` carries an all-zero commit id: it is a placeholder, not a real upstream revision. The real
corpus pin is written at image bake time and the adapter refuses to run without it.

`.meta/example.py` exists in one exercise purely to prove that the adapter never shows `.meta` to the
model and never stages it for the worker.

`conftest.py` keeps this whole tree out of pytest collection: the exercise test files are named
`*_test.py` like upstream, and their stubs fail by construction, so they must never run as part of this
repository's own suite. They are data, read as text by `tests/test_benchmarks_aider_polyglot.py`.

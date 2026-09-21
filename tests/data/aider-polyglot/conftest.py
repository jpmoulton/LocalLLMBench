"""Keep this corpus out of pytest collection.

The hand-made exercises mirror the upstream layout, so their test files are named ``*_test.py`` and
pytest's default ``python_files`` would otherwise collect them as part of this repository's own suite.
They are data: they describe an exercise for a model to solve, they are never run here, and their stubs
fail by construction. ``test_benchmarks_aider_polyglot.py`` reads them as text.
"""

collect_ignore_glob = ["*"]

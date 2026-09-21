import pytest

from llmbench.config import BackendSettings, ExperimentManifest, ModelArtifact, TaskSelection


@pytest.fixture
def manifest():
    return ExperimentManifest(
        model=ModelArtifact(model_key="synthetic-fixture", sha256="0" * 64, source_revision="fixture-v1",
                            quantization="synthetic", tokenizer_hash="fixture-v1", template_hash="fixture-v1",
                            provenance={"source": {"revisions": ["v1"]}}),
        backend=BackendSettings(engine="mock", runtime_revision="mock-v1", context_length=8192),
        tasks=(TaskSelection(suite="offline-demo", revision="v1", task_ids=("tool-1", "needle-1")),),
        harness_revision="0.1.0", scorer_revision="v1")

# LocalLLMBench

LocalLLMBench compares local model and serving-configuration combinations using repeatable benchmarks. Its results inform the user's quality and performance tradeoffs.

## Language

**Run**:
The complete end-to-end workflow evaluating multiple model and serving-configuration combinations under fixed benchmark protocols and answer-generation settings. Results describe those combinations without isolating the cause of a difference when several settings change together.
_Avoid_: Sweep (as a competing name for the complete execution); run (for an individual model/configuration evaluation within it)

**Recipe**:
A reusable specification of the models, serving-configuration choices, benchmarks, and fixed evaluation settings for a run. A recipe describes the requested experiment, not its execution or results.

**Trial**:
One execution of one model with one serving configuration against all benchmarks selected for a run. Separate executions of the same model/configuration combination are separate trials.
_Avoid_: Run (for this nested unit)

**Benchmark evaluation**:
One benchmark's execution and results within a trial.
_Avoid_: Trial (for an individual benchmark's execution)

**Serving configuration**:
The settings governing how an inference server loads and serves a model, including context capacity, cache precision, and offloading settings.
_Avoid_: Model (when referring to serving settings rather than the model itself)

**Benchmark protocol**:
The specification of a benchmark's selected problems, prompt construction, and scoring rules. The protocol is held constant across the model and serving-configuration combinations evaluated in a run.

**Answer-generation settings**:
The sampling parameters, generation seed, reasoning mode, and output-token budget governing model responses. These settings remain fixed across model and serving-configuration combinations within a run.
_Avoid_: Serving configuration (when referring to request-level response-generation controls)

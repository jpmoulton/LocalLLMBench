"""Sequential host orchestration. Only run/trial-labelled containers are cleaned up."""

import copy
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

import benchmark
import inference
from orchestrator.logging import RunLogs
from orchestrator.report import write_results, write_status
from orchestrator.state import now, save_json, update_counts


class DockerBackend:
    """Docker CLI and HTTP boundary; no model calls occur during construction."""

    def __init__(self, *, logs: RunLogs | None = None):
        self.log = logs if logs is not None else RunLogs()
        self._log_process = None
        self._log_reader = None
        self._log_error = None

    def _follow_logs(self, name: str) -> None:
        process = subprocess.Popen(
            ["docker", "logs", "--follow", "--timestamps", name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        self._log_process = process

        self._log_error = None

        def collect():
            try:
                for line in process.stdout:
                    self.log.emit("llama.cpp", line)
            except Exception as exc:
                self._log_error = exc

        self._log_reader = threading.Thread(target=collect, daemon=True)
        self._log_reader.start()

    def _stop_log_stream(self) -> None:
        process = self._log_process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    process.wait(timeout=2)  # Let the stopped container's final output drain.
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            if self._log_reader is not None:
                self._log_reader.join(timeout=5)
        finally:
            process.stdout.close()
            self._log_process = self._log_reader = None
        if self._log_error is not None:
            raise RuntimeError(f"Inference log stream failed: {self._log_error}") from self._log_error

    def command(self, args: list[str], *, check: bool = True, timeout: float = 60):
        result = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        if check and result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip() or f"Command failed: {args[:3]}")
        return result

    def docker_json(self, *args: str):
        return json.loads(self.command(["docker", *args]).stdout)

    def ensure_network(self) -> None:
        result = self.command(["docker", "network", "inspect", benchmark.NETWORK], check=False)
        if result.returncode:
            self.command(["docker", "network", "create", benchmark.NETWORK])

    def start(self, trial: dict, plan: dict, name: str, labels: dict) -> dict:
        self.command(inference.build_command(
            Path(trial["model"]), name=name, port=plan["inference"]["port"],
            bind_address=plan["inference"]["bind_address"], server_args=trial["server_args"],
            labels=labels, network=benchmark.NETWORK,
        ))
        self._follow_logs(name)
        details = self.docker_json("inspect", name)[0]
        return {
            "container": name, "image_id": details["Image"],
            "image": self.docker_json("image", "inspect", details["Image"])[0],
            "command": details["Config"]["Cmd"], "environment": details["Config"]["Env"],
            "mounts": details["Mounts"],
        }

    def wait_ready(self, name: str, base_url: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            details = self.docker_json("inspect", name)[0]
            if not details["State"]["Running"]:
                raise RuntimeError(f"Inference container exited during startup: {name}")
            try:
                with urllib.request.urlopen(base_url + "/health", timeout=min(2, max(.1, deadline - time.monotonic()))) as response:
                    healthy = json.load(response).get("status") == "ok"
                if healthy:
                    # Never probe a different listener if the container failed while health was in flight.
                    if not self.docker_json("inspect", name)[0]["State"]["Running"]:
                        raise RuntimeError(f"Inference container exited during startup: {name}")
                    return
            except urllib.error.HTTPError as exc:
                # urlopen raises before entering the response context (e.g. HTTP 503).
                exc.close()
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                # A published port can accept then drop connections before the server is ready.
                # Keep polling within the startup deadline; never retry generation here.
                pass
            time.sleep(min(.5, max(0, deadline - time.monotonic())))
        raise TimeoutError(f"Inference readiness timed out after {timeout:g}s: {name}")

    def probe(self, base_url: str, timeout: float) -> dict:
        body = {
            "model": "local-model", "messages": [{"role": "user", "content": "Say Hello world."}],
            "max_tokens": 32, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        return {"request": body, "response": result, "latency_seconds": time.monotonic() - started}

    def run_benchmarks(self, command: list[str], tick) -> int:
        """Save controller stdout/stderr without forwarding it to the main terminal."""
        messages = queue.Queue()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

        def collect():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        heartbeat = time.monotonic()
        try:
            while True:
                try:
                    line = messages.get(timeout=1)
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                if line:
                    self.log.emit("benchmarks", line)
                if time.monotonic() - heartbeat >= 15:
                    tick()
                    heartbeat = time.monotonic()
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            reader.join(timeout=5)
            process.stdout.close()


    def cleanup(self, labels: dict) -> None:
        try:
            filters = [part for key, value in labels.items() for part in ("--filter", f"label={key}={value}")]
            ids = self.command(["docker", "ps", "--all", "--quiet", *filters]).stdout.split()
            if ids:
                try:
                    self.command(["docker", "stop", "--time", "5", *ids], check=False, timeout=10 + 5 * len(ids))
                except subprocess.TimeoutExpired:
                    pass  # Force removal below is cleanup, not a trial retry.
            # Re-query after stopping controllers so their last-created sandboxes are included.
            ids = self.command(["docker", "ps", "--all", "--quiet", *filters]).stdout.split()
            if ids:
                self.command(["docker", "rm", "--force", "--volumes", *ids])
            if self.command(["docker", "ps", "--all", "--quiet", *filters]).stdout.strip():
                raise RuntimeError("Owned trial containers remain after cleanup; refusing to start another trial")
        finally:
            self._stop_log_stream()


def server_url(plan: dict) -> str:
    address = plan["inference"]["bind_address"].strip("[]")
    address = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(address, address)
    if ":" in address:
        address = f"[{address}]"
    return f"http://{address}:{plan['inference']['port']}"


def _error(phase: str, exc: BaseException) -> dict:
    return {"phase": phase, "type": type(exc).__name__, "message": str(exc), "at": now()}


def _show_results(directory: Path, plan: dict, logs: RunLogs) -> None:
    for name in plan["benchmarks"]:
        path = directory / name / "summary.json"
        if not path.is_file():
            logs.print(f"  {name}: no benchmark summary")
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            metrics = {key: result.get(key) for key in result.get("score_metrics", {})}
            logs.print(f"  {name}: {result.get('status', 'unknown')} | scores: {json.dumps(metrics)}")
            performance = result.get("performance", {})
            fields = (("decode_tokens_per_second", "decode tok/s"),
                      ("decode_ms_per_token", "decode ms/token"),
                      ("prompt_tokens_per_second", "prompt tok/s"),
                      ("request_latency_seconds_mean", "mean request seconds"))
            values = [f"{label}: {performance[key]:.3f}" for key, label in fields
                      if isinstance(performance.get(key), (int, float)) and not isinstance(performance[key], bool)]
            logs.print("    " + (" | ".join(values) if values else "Generation timing unavailable"))
        except (OSError, ValueError, TypeError) as exc:
            logs.print(f"  {name}: could not display summary: {exc}")


def execute(
    plan: dict, *, backend=None, reporter=write_results, logs: RunLogs | None = None,
) -> tuple[Path, dict, dict | None]:
    if logs is None:
        with RunLogs() as owned_logs:
            return execute(plan, backend=backend, reporter=reporter, logs=owned_logs)
    backend = backend if backend is not None else DockerBackend(logs=logs)
    run_id = now().replace("-", "").replace(":", "").split(".")[0] + "-" + uuid.uuid4().hex[:8]
    output = Path(plan["output_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    logs.open_run(output)
    started = time.monotonic()
    run = {
        "schema_version": 1, "run_id": run_id, "name": plan["name"], "status": "running",
        "phase": "starting", "started_at": now(), "finished_at": None, "elapsed_seconds": 0,
        "current_trial": None, "errors": [], "trials": [
            {**copy.deepcopy(trial), "status": "pending", "phase": "pending", "started_at": None,
             "finished_at": None, "duration_seconds": None,
             "durations": {key: None for key in ("load", "probe", "benchmark", "cleanup")},
             "error": None, "benchmark_summary": f"trials/{trial['id']}/benchmarks/summary.json"}
            for trial in plan["trials"]
        ],
    }
    save_json(output / "execution_plan.json", plan)
    report = None

    def persist(trial=None):
        run["elapsed_seconds"] = round(time.monotonic() - started, 3)
        update_counts(run)
        if trial is not None:
            save_json(output / "trials" / trial["id"] / "trial.json", trial)
        write_status(output, run)

    def report_results():
        nonlocal report
        try:
            report = reporter(output, plan, run)
        except Exception as exc:
            failure = _error("report", exc)
            if not any(item["phase"] == "report" and item["message"] == failure["message"] for item in run["errors"]):
                run["errors"].append(failure)
            logs.print(f"Report export failed; JSON results remain available: {exc}")
            persist()

    persist()
    logs.print(f"Run output: {output}")
    logs.print(f"Live logs (append across runs): {logs.live_dir}")
    logs.print(f"This run's logs: {output / 'logging'}")
    interrupted = False
    try:
        backend.ensure_network()
        for index, trial in enumerate(run["trials"], 1):
            logs.trial = trial["id"]
            directory = output / "trials" / trial["id"]
            directory.mkdir(parents=True, exist_ok=True)
            labels = {"localllmbench.run": run_id, "localllmbench.trial": trial["id"]}
            inference_name = f"llmb-{run_id}-{trial['id']}-inference".lower()
            benchmark_name = f"llmb-{run_id}-{trial['id']}-benchmark".lower()
            trial_started = time.monotonic()
            trial.update(status="running", phase="load", started_at=now())
            run.update(current_trial=trial["id"], phase="load")
            persist(trial)
            logs.print(f"\nTrial {index}/{len(run['trials'])}: {Path(trial['model']).name} | {trial['group']}")
            logs.print(f"  Parameters: {json.dumps(trial['parameters'], sort_keys=True)}")
            phase_started = time.monotonic()
            cleanup_failed = False
            try:
                metadata = backend.start(trial, plan, inference_name, labels)
                save_json(directory / "inference.json", metadata)
                backend.wait_ready(inference_name, server_url(plan), plan["execution"]["startup_timeout"])
                trial["durations"]["load"] = time.monotonic() - phase_started
                trial["phase"] = run["phase"] = "probe"
                phase_started = time.monotonic()
                persist(trial)
                probe = backend.probe(server_url(plan), plan["execution"]["request_timeout"])
                save_json(directory / "probe.json", probe)
                choices = probe.get("response", {}).get("choices", [])
                content = choices[0].get("message", {}).get("content") if choices else None
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Hello world probe returned no assistant text")
                trial["durations"]["probe"] = time.monotonic() - phase_started
                logs.print(f"  Probe: {content.strip()}")
                trial["phase"] = run["phase"] = "benchmark"
                phase_started = time.monotonic()
                persist(trial)
                config = {
                    "benchmarks": plan["benchmarks"], "benchmark_options": plan["benchmark_options"],
                    "generation": plan["generation"], "output_dir": "/results",
                    "execution": {key: plan["execution"][key] for key in ("concurrency", "request_timeout")},
                    "inference": {"container": inference_name, "model": "local-model"},
                }
                benchmark_output = directory / "benchmarks"
                command = benchmark.build_command(
                    config, benchmark_output, name=benchmark_name, labels=labels,
                    expected_manifests=plan["manifests"],
                )

                def tick():
                    persist(trial)
                    logs.print(f"  Trial {index}/{len(run['trials'])} running: {time.monotonic() - trial_started:.1f}s; "
                               f"run elapsed {run['elapsed_seconds']:.1f}s")

                code = backend.run_benchmarks(command, tick)
                trial["durations"]["benchmark"] = time.monotonic() - phase_started
                summary_path = benchmark_output / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
                if code != 0 or summary.get("status") != "completed":
                    raise RuntimeError(f"Benchmark controller exited {code}; suite status {summary.get('status', 'missing')}")
                trial["status"] = "completed"
            except KeyboardInterrupt as exc:
                interrupted = True
                trial["status"] = "cancelled"
                trial["error"] = _error(trial["phase"], exc)
                logs.print("\nInterrupted. Cleaning up this trial's containers.")
            except Exception as exc:
                trial["status"] = "incomplete"
                trial["error"] = _error(trial["phase"], exc)
                logs.print(f"  Trial incomplete during {trial['phase']}: {exc}")
            finally:
                previous_phase = trial["phase"]
                if previous_phase in trial["durations"] and trial["durations"][previous_phase] is None:
                    trial["durations"][previous_phase] = time.monotonic() - phase_started
                trial["phase"] = run["phase"] = "cleanup"
                def stop_after_cleanup(signum, frame):
                    nonlocal interrupted
                    interrupted = True

                previous_handler = None
                if threading.current_thread() is threading.main_thread():
                    previous_handler = signal.signal(signal.SIGINT, stop_after_cleanup)
                cleanup_started = time.monotonic()
                try:
                    try:
                        persist(trial)
                    except Exception as exc:
                        run["errors"].append(_error("status_write", exc))
                    try:
                        backend.cleanup(labels)
                    except Exception as exc:
                        cleanup_failed = True
                        trial["status"] = "incomplete"
                        run["errors"].append(_error("cleanup", exc))
                        logs.print(f"  Cleanup failed; stopping the run rather than overlapping containers: {exc}")
                finally:
                    if previous_handler is not None:
                        signal.signal(signal.SIGINT, previous_handler)
                trial["durations"]["cleanup"] = time.monotonic() - cleanup_started
                trial["duration_seconds"] = time.monotonic() - trial_started
                trial["finished_at"] = now()
                trial["phase"] = "finished"
                run["current_trial"] = None
                persist(trial)
            _show_results(directory / "benchmarks", plan, logs)
            logs.print(f"  {trial['status']}; trial {trial['duration_seconds']:.1f}s; "
                       f"{len(run['trials']) - index} remaining; run elapsed {run['elapsed_seconds']:.1f}s")
            report_results()
            if interrupted or cleanup_failed:
                break
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        run["errors"].append(_error(run["phase"], exc))
        logs.print(f"Run stopped: {exc}")
    finally:
        logs.trial = None
        run["status"] = "cancelled" if interrupted else (
            "completed" if all(trial["status"] == "completed" for trial in run["trials"]) and not run["errors"]
            else "incomplete"
        )
        run.update(phase="finished", current_trial=None, finished_at=now())
        persist()
        report_results()
        if run["errors"] and run["status"] == "completed":
            run["status"] = "incomplete"
            persist()
    return output, run, report

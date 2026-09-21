"""Optional explicit telemetry; importing this module never probes hardware."""

import json
import subprocess
import time


def parse_nvidia_csv(text: str) -> list[dict]:
    fields = ("index", "name", "memory_used_mib", "memory_total_mib", "gpu_utilization_percent",
              "power_watts", "temperature_celsius")
    result = []
    for line in text.strip().splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(fields):
            raise ValueError("unexpected nvidia-smi field count")
        row = dict(zip(fields, values))
        for key in fields:
            if key != "name":
                try:
                    row[key] = float(row[key])
                except ValueError:
                    row[key] = None
        result.append(row)
    return result


def sample_system(*, include_gpu=False, command_runner=subprocess.run) -> dict:
    import psutil
    memory = psutil.virtual_memory()
    row = {"monotonic_seconds": time.monotonic(), "host_memory_used_bytes": memory.used,
           "host_memory_available_bytes": memory.available, "host_cpu_percent": psutil.cpu_percent(),
           "gpus": [], "gpu_error": None}
    if include_gpu:
        try:
            completed = command_runner([
                "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
            row["gpus"] = parse_nvidia_csv(completed.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            row["gpu_error"] = str(exc)
    # nvidia-smi memory is not a measurement of Windows shared GPU memory.
    row["shared_gpu_memory_bytes"] = None
    json.dumps(row, allow_nan=False)
    return row

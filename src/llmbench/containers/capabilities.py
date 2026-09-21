"""Capability table parsed from the pinned llama-server --help capture. Pure; no process is started."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ..backends.base import UnsupportedSetting

HELP_FILE = "llama-server-help.txt"
VERSION_FILE = "llama-server-version.txt"

# Exactly the flags plan.build_server_argv may emit.
ALLOWED_FLAGS = frozenset({
    "--model", "--host", "--port", "--alias", "--ctx-size", "--n-gpu-layers", "--cache-type-k", "--cache-type-v",
    "--kv-offload", "--no-kv-offload", "--flash-attn", "--load-mode", "--batch-size", "--ubatch-size",
    "--threads", "--threads-batch", "--parallel", "--cache-prompt", "--no-cache-prompt", "--cache-reuse",
    "--cache-ram", "--no-context-shift", "--fit", "--swa-full", "--jinja", "--reasoning", "--reasoning-format",
    "--reasoning-budget", "--spec-type", "--spec-draft-n-max", "--warmup", "--no-warmup", "--no-ui", "--metrics",
    "--slots", "--offline", "--log-verbosity"})
# Never emitted: tools/agents, downloads, routers, mutable properties, secrets, unverified model overrides.
DENIED_FLAGS = ("--tools", "--tools-*", "--agent", "--ui-mcp-proxy", "--mcp-servers-*", "-hf*", "--hf-*",
                "--model-url", "--docker-repo", "--models-*", "--props", "--api-key*", "--lora*", "--override-*",
                "--chat-template*", "--grammar*", "--log-file", "--slot-save-path", "--media-path",
                "--spec-default")


@dataclass(frozen=True)
class FlagSpec:
    names: tuple[str, ...]
    value_hint: str | None
    choices: tuple[str, ...]
    choices_closed: bool
    default: str | None
    env: str | None
    removed: bool
    deprecated: bool
    section: str | None


@dataclass(frozen=True)
class ServerCapabilities:
    flags: Mapping[str, FlagSpec]
    help_sha256: str
    version: str | None = None
    build: str | None = None


def normalize_help(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip() + "\n"


def help_sha256(text: str) -> str:
    return hashlib.sha256(normalize_help(text).encode("utf-8")).hexdigest()


def parse_version(text: str) -> dict:
    match = re.search(r"version: (\S+) \(build (\d+), commit ([0-9a-f]+)\)", text)
    if match is None:
        raise ValueError("unrecognized llama-server --version output")
    return {"version": match[1], "build": int(match[2]), "commit": match[3],
            "build_info": f"b{match[2]}-{match[3]}"}


def _choices(hint: str | None, lines: list[str]) -> tuple[tuple[str, ...], bool]:
    hint = hint or ""
    for pattern, separator in ((r"\[([^\[\]|]+(?:\|[^\[\]|]+)+)\]", "|"), (r"\{([^{},]+(?:,[^{},]+)+)\}", ","),
                               (r"<([^<>|.]+(?:\|[^<>|.]+)+)>", "|"), (r"([a-z0-9_-]+(?:,[a-z0-9_-]+)+)", ",")):
        match = re.fullmatch(pattern, hint)
        if match:
            return tuple(match[1].split(separator)), True
    for line in lines:
        if line.startswith("allowed values:"):
            return tuple(item.strip() for item in line.removeprefix("allowed values:").split(",")), True
    return tuple(match[1] for line in lines if (match := re.match(r"- ([^\s:]+):", line))), False


def _spec(head: str, lines: list[str], section: str | None) -> FlagSpec:
    tokens = head.split(", ")
    name, _, hint = tokens[-1].partition(" ")
    names = (*tokens[:-1], name)
    if not all(re.fullmatch(r"--?[A-Za-z0-9][A-Za-z0-9._-]*", item) for item in names):
        raise ValueError(f"unparseable help entry: {head!r}")
    text = " ".join(lines)
    choices, closed = _choices(hint or None, lines)
    default = re.search(r"default: '?([^',)]*)", text)
    env = re.search(r"\(env: ([A-Z0-9_]+)\)", text)
    return FlagSpec(names, hint or None, choices, closed, default[1].strip() if default else None,
                    env[1] if env else None, "has been removed" in text, "deprecated" in text.lower(), section)


def parse_server_help(text: str, *, version_text: str | None = None) -> ServerCapabilities:
    section, entries = None, []
    current = None
    for line in normalize_help(text).split("\n"):
        header = re.fullmatch(r"-{5} (.+) -{5}", line)
        if header:
            section, current = header[1], None
        elif line.startswith("-"):
            parts = re.split(r"\s{2,}", re.sub(r",\s+", ", ", line), maxsplit=1)
            current = (parts[0], [parts[1]] if len(parts) > 1 else [], section)
            entries.append(current)
        elif line[:1].isspace() and current is not None:
            if line.strip():
                current[1].append(line.strip())
        elif line:
            current = None  # Preamble or diagnostics outside the option table.
    flags: dict[str, FlagSpec] = {}
    for head, lines, where in entries:
        spec = _spec(head, lines, where)
        for name in spec.names:
            if name in flags:
                raise ValueError(f"duplicate help flag {name}")
            flags[name] = spec
    if not flags:
        raise ValueError("help text contains no options")
    version = parse_version(version_text) if version_text else {}
    return ServerCapabilities(MappingProxyType(flags), help_sha256(text), version.get("version"),
                              version.get("build_info"))


def load_capabilities(directory: str | Path) -> ServerCapabilities:
    root = Path(directory)
    version = root / VERSION_FILE
    return parse_server_help((root / HELP_FILE).read_text(encoding="utf-8-sig"),
                             version_text=version.read_text(encoding="utf-8-sig") if version.exists() else None)


def _denied(names: tuple[str, ...]) -> bool:
    return any(fnmatchcase(name, pattern) for name in names for pattern in DENIED_FLAGS)


def check_argv(argv, caps: ServerCapabilities) -> list[str]:
    findings, index, items = [], 0, list(argv)
    while index < len(items):
        token = items[index]
        index += 1
        if type(token) is not str or not token.startswith("-"):
            findings.append(f"unexpected positional argument {token!r}")
            continue
        spec = caps.flags.get(token)
        if _denied(spec.names if spec else (token,)):
            findings.append(f"{token} is denied by the harness")
        if token not in ALLOWED_FLAGS:
            findings.append(f"{token} is outside the harness allowlist")
        if spec is None:
            findings.append(f"{token} is unknown to this llama-server build")
            continue
        if spec.removed:
            findings.append(f"{token} has been removed from this llama-server build")
        if spec.value_hint is None:
            if index < len(items) and type(items[index]) is str and not items[index].startswith("-"):
                findings.append(f"boolean flag {token} was given a value")
                index += 1
            continue
        if index >= len(items) or items[index] in caps.flags:
            findings.append(f"{token} requires a value")
            continue
        value = items[index]
        index += 1
        parts = value.split(",") if re.fullmatch(r"[a-z0-9_-]+(?:,[a-z0-9_-]+)+", spec.value_hint) else [value]
        if spec.choices_closed and any(part not in spec.choices for part in parts):
            findings.append(f"{token} value {value!r} is outside {list(spec.choices)}")
    return findings


def require_supported(argv, caps: ServerCapabilities, *, expected_help_sha256: str) -> None:
    if caps.help_sha256 != expected_help_sha256:
        raise UnsupportedSetting("saved llama-server help does not match the configured inference image help_sha256")
    findings = check_argv(argv, caps)
    if findings:
        raise UnsupportedSetting("unsupported llama-server argv: " + "; ".join(findings))

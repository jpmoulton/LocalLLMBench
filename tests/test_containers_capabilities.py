from fnmatch import fnmatchcase
from pathlib import Path

import pytest

from llmbench.backends.base import UnsupportedSetting
from llmbench.containers.capabilities import (ALLOWED_FLAGS, DENIED_FLAGS, check_argv, help_sha256,
                                              load_capabilities, normalize_help, parse_server_help,
                                              parse_version, require_supported)
from llmbench.containers.config import read_run_config
from llmbench.containers.plan import build_server_argv

DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
HELP = (DATA / "llama-server-help-b11011.txt").read_bytes().decode("utf-8")
VERSION = (DATA / "llama-server-version-b11011.txt").read_text(encoding="utf-8")
CAPS = parse_server_help(HELP, version_text=VERSION)


def test_real_help_parses_with_version():
    # The capture was taken on Windows (CRLF); a git checkout hands it to Linux and macOS as LF. Parsing and the
    # pinned hash must not depend on which one is on disk, so both renderings are exercised explicitly.
    unix = HELP.replace("\r\n", "\n")
    windows = unix.replace("\n", "\r\n")
    for rendering in (unix, windows):
        assert len(parse_server_help(rendering, version_text=VERSION).flags) == 411
    assert help_sha256(unix) == help_sha256(windows)
    assert len(CAPS.flags) == 411
    assert CAPS.version == "0.4.1-dev" and CAPS.build == "b11011-aa39d7a3e"
    assert parse_version(VERSION) == {"version": "0.4.1-dev", "build": 11011, "commit": "aa39d7a3e",
                                      "build_info": "b11011-aa39d7a3e"}
    assert CAPS.flags["-ngl"] is CAPS.flags["--n-gpu-layers"]
    assert CAPS.flags["--no-ui"].names == ("--ui", "--webui", "--no-ui", "--no-webui")
    assert CAPS.flags["--threads"].env == "LLAMA_ARG_THREADS" and CAPS.flags["--threads"].default == "-1"
    assert CAPS.flags["--ctx-size"].section == "common params"
    assert CAPS.flags["--spec-draft-n-max"].section == "speculative params"
    with pytest.raises(ValueError):
        parse_version("llama-server unknown")
    with pytest.raises(ValueError):
        parse_server_help("no options here\n")


def test_closed_and_advisory_choices():
    closed = {"--flash-attn": ("on", "off", "auto"), "--fit": ("on", "off"), "--cpu-strict": ("0", "1"),
              "--rope-scaling": ("none", "linear", "yarn"), "--reasoning": ("on", "off", "auto")}
    for name, choices in closed.items():
        assert CAPS.flags[name].choices == choices and CAPS.flags[name].choices_closed
    assert CAPS.flags["--cache-type-v"].choices[:4] == ("f32", "f16", "bf16", "q8_0")
    assert CAPS.flags["--cache-type-v"].choices_closed
    assert "draft-mtp" in CAPS.flags["--spec-type"].choices and CAPS.flags["--spec-type"].choices_closed
    load_mode = CAPS.flags["--load-mode"]
    assert {"none", "mmap", "dio"} <= set(load_mode.choices) and not load_mode.choices_closed
    assert CAPS.flags["--reasoning-format"].choices == ("none", "deepseek", "deepseek-legacy")
    for name in ("--tensor-split", "--poll", "--device", "--override-tensor", "--threads"):
        assert not CAPS.flags[name].choices_closed
    assert CAPS.flags["--kv-offload"].value_hint is None and CAPS.flags["--ctx-size"].value_hint == "N"


def test_removed_and_deprecated_flags():
    assert CAPS.flags["--draft-max"].removed and CAPS.flags["--spec-ngram-size-n"].removed
    assert CAPS.flags["--defrag-thold"].deprecated and not CAPS.flags["--ctx-size"].removed
    assert "--no-mmap" not in CAPS.flags


def test_hash_is_newline_and_trailing_space_insensitive():
    unix = HELP.replace("\r\n", "\n")
    assert help_sha256(HELP) == help_sha256(unix) == help_sha256(unix.replace("\n", "   \n") + "\n\n")
    assert normalize_help(HELP).endswith("--spec-default                          enable default speculative "
                                        "decoding config\n")
    assert help_sha256(unix.replace("--fit ", "--fat ")) != help_sha256(HELP)


def test_allowed_flags_exist_and_never_overlap_denied():
    assert ALLOWED_FLAGS <= set(CAPS.flags)
    assert not [flag for flag in ALLOWED_FLAGS if any(fnmatchcase(flag, pattern) for pattern in DENIED_FLAGS)]
    assert not [flag for flag in ALLOWED_FLAGS if CAPS.flags[flag].removed or CAPS.flags[flag].deprecated]
    assert "--log-colors" not in ALLOWED_FLAGS and "--no-webui" not in ALLOWED_FLAGS


@pytest.mark.parametrize("name", ["candidate.json", "candidate-mtp.json"])
def test_example_argv_is_supported(name):
    config = read_run_config(EXAMPLES / name)
    argv = build_server_argv(config)
    assert {item for item in argv if item.startswith("--")} <= ALLOWED_FLAGS
    assert check_argv(argv, CAPS) == []
    require_supported(argv, CAPS, expected_help_sha256=config.inference_image.help_sha256)


@pytest.mark.parametrize("argv, fragment", [
    (("--tools", "all"), "denied"), (("-hf", "org/model"), "denied"), (("--no-agent",), "denied"),
    (("--hf-repo", "org/model"), "denied"), (("--api-key", "secret"), "denied"),
    (("--chat-template-file", "/x"), "denied"), (("--lora", "/x"), "denied"),
    (("--seed", "1"), "allowlist"), (("--no-mmap",), "unknown"), (("--draft-max", "4"), "removed"),
    (("--jinja", "yes"), "given a value"), (("--ctx-size",), "requires a value"),
    (("--ctx-size", "--jinja"), "requires a value"), (("--flash-attn", "maybe"), "outside"),
    (("--spec-type", "draft-mtp,bogus"), "outside"), (("--cache-type-k", "q2_k"), "outside"),
    (("model.gguf",), "positional"),
])
def test_bad_argv_findings(argv, fragment):
    findings = check_argv(argv, CAPS)
    assert any(fragment in finding for finding in findings), findings
    with pytest.raises(UnsupportedSetting):
        require_supported(argv, CAPS, expected_help_sha256=CAPS.help_sha256)


def test_negative_numbers_and_multi_choice_values_are_values():
    assert check_argv(("--reasoning-budget", "-1", "--spec-type", "none"), CAPS) == []


def test_mutated_help_and_hash_mismatch_reject():
    config = read_run_config(EXAMPLES / "candidate.json")
    argv = build_server_argv(config)
    mutated = parse_server_help(HELP.replace("-fit,  --fit [on|off]", "-fit,  --fit-removed [on|off]"))
    with pytest.raises(UnsupportedSetting, match="--fit is unknown"):
        require_supported(argv, mutated, expected_help_sha256=mutated.help_sha256)
    with pytest.raises(UnsupportedSetting, match="help_sha256"):
        require_supported(argv, mutated, expected_help_sha256=config.inference_image.help_sha256)
    narrowed = parse_server_help(HELP.replace("--flash-attn [on|off|auto]", "--flash-attn [off|auto]    "))
    with pytest.raises(UnsupportedSetting, match="--flash-attn value 'on'"):
        require_supported(argv, narrowed, expected_help_sha256=narrowed.help_sha256)


def test_load_capabilities_reads_saved_preparation_files(tmp_path):
    (tmp_path / "llama-server-help.txt").write_bytes(HELP.encode("utf-8"))
    assert load_capabilities(tmp_path).version is None
    (tmp_path / "llama-server-version.txt").write_text(VERSION, encoding="utf-8")
    loaded = load_capabilities(tmp_path)
    assert loaded.help_sha256 == CAPS.help_sha256 and loaded.build == "b11011-aa39d7a3e"
    with pytest.raises(OSError):
        load_capabilities(tmp_path / "missing")

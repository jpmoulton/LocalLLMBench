"""GGUF header reader: synthetic v2/v3 fixtures built here; no model file is ever opened."""

import hashlib
import json
import struct
from pathlib import Path

import pytest

from llmbench.containers.gguf import (GGUFError, gguf_summary, parse_gguf_header, read_gguf_metadata,
                                      template_hash)

DATA = Path(__file__).parent / "data"
PILOT_TEMPLATE_ID = "sha256:c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(kind: str, value) -> bytes:
    if kind == "str":
        return struct.pack("<I", 8) + _string(value)
    if kind == "u32":
        return struct.pack("<I", 4) + struct.pack("<I", value)
    if kind == "u64":
        return struct.pack("<I", 10) + struct.pack("<Q", value)
    if kind == "f32":
        return struct.pack("<I", 6) + struct.pack("<f", value)
    if kind == "bool":
        return struct.pack("<I", 7) + struct.pack("<?", value)
    if kind == "arr-str":
        return struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + b"".join(map(_string, value))
    if kind == "arr-f32":
        return struct.pack("<I", 9) + struct.pack("<I", 6) + struct.pack("<Q", len(value)) + struct.pack(
            f"<{len(value)}f", *value)
    if kind == "arr-arr-u32":
        inner = b"".join(struct.pack("<I", 4) + struct.pack("<Q", len(row)) + struct.pack(f"<{len(row)}I", *row)
                         for row in value)
        return struct.pack("<I", 9) + struct.pack("<I", 9) + struct.pack("<Q", len(value)) + inner
    raise AssertionError(kind)


def build_gguf(entries, *, version=3, tensors=0) -> bytes:
    body = b"".join(_string(key) + _value(kind, value) for key, kind, value in entries)
    return b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", tensors) + struct.pack("<Q", len(entries)) + body


def pilot_template() -> str:
    return json.loads((DATA / "props-b11011.json").read_text(encoding="utf-8"))["body"]["chat_template"]


def fixture_entries(template="{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"):
    tokens = [f"tok{index}" for index in range(5000)]  # a vocabulary-sized array the reader must skip
    return [
        ("general.architecture", "str", "qwen3next"),
        ("general.name", "str", "Qwen3.8 27B"),
        ("general.file_type", "u32", 15),
        ("qwen3next.context_length", "u32", 262144),
        ("qwen3next.block_count", "u32", 48),
        ("qwen3next.nextn_predict_layers", "u32", 1),
        ("qwen3next.rope.freq_base", "f32", 10000000.0),
        ("tokenizer.ggml.tokens", "arr-str", tokens),
        ("tokenizer.ggml.scores", "arr-f32", [float(index) for index in range(5000)]),
        ("tokenizer.ggml.nested", "arr-arr-u32", [[1, 2, 3], [4], []]),
        ("tokenizer.ggml.add_bos_token", "bool", False),
        ("tokenizer.chat_template", "str", template),
        ("general.quantization_version", "u32", 2),
    ]


def test_reads_chat_template_from_v3_header_fixture(tmp_path):
    template = "{{ bos_token }}{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    path = tmp_path / "model-Q4_K_M.gguf"
    path.write_bytes(build_gguf(fixture_entries(template), tensors=7) + b"\0" * 4096)  # tensor info/data follows
    found = read_gguf_metadata(path)
    assert found["tokenizer.chat_template"] == template
    assert found["general.architecture"] == "qwen3next" and found["general.name"] == "Qwen3.8 27B"
    # the fixture declares 7 tensors but carries no tensor-info block: the types are unknown, not guessed
    assert found["__gguf__"] == {"version": 3, "tensor_count": 7, "kv_count": 13, "tensor_types": None}
    assert "tokenizer.ggml.tokens" not in found and "general.file_type" not in found  # arrays skipped, keys filtered
    everything = read_gguf_metadata(path, keys=None)
    assert everything["general.file_type"] == 15 and everything["qwen3next.context_length"] == 262144
    assert everything["tokenizer.ggml.add_bos_token"] is False
    assert everything["qwen3next.rope.freq_base"] == pytest.approx(10000000.0)
    assert not any(key.startswith("tokenizer.ggml.") and key != "tokenizer.ggml.add_bos_token" for key in everything)
    summary = gguf_summary(path)
    assert summary["template_hash"] == "sha256:" + hashlib.sha256(template.encode()).hexdigest()
    assert summary["n_ctx_train"] == 262144 and summary["nextn_predict_layers"] == 1 and summary["block_count"] == 48
    assert summary["quantization"] == "Q4_K_M" and summary["file_type"] == 15 and summary["name"] == "Qwen3.8 27B"
    # v2 uses the same layout for everything this reader touches.
    v2 = build_gguf(fixture_entries(template), version=2)
    assert parse_gguf_header(v2)["tokenizer.chat_template"] == template
    assert parse_gguf_header(v2)["__gguf__"]["version"] == 2


def test_rejects_bad_magic_oversize_header_and_truncation(tmp_path):
    good = build_gguf(fixture_entries())
    with pytest.raises(GGUFError, match="magic"):
        parse_gguf_header(b"GGML" + good[4:])
    with pytest.raises(GGUFError, match="version"):
        parse_gguf_header(build_gguf(fixture_entries(), version=1))
    with pytest.raises(GGUFError, match="version"):
        parse_gguf_header(build_gguf(fixture_entries(), version=4))
    # Truncated anywhere inside the header is an error, never a partial answer.
    for cut in (3, 12, 40, len(good) // 2, len(good) - 1):
        with pytest.raises(GGUFError, match="truncated|bound"):
            parse_gguf_header(good[:cut], keys=None)
    # With a key filter the reader stops once every requested key is known: bytes after that are not read.
    assert parse_gguf_header(good[:-1])["tokenizer.chat_template"]
    with pytest.raises(GGUFError, match="truncated|bound"):
        parse_gguf_header(good[:-1], keys=("general.quantization_version",))
    # A header larger than the bounded read fails closed with a distinct reason.
    path = tmp_path / "big.gguf"
    path.write_bytes(good)
    with pytest.raises(GGUFError, match="bound"):
        read_gguf_metadata(path, max_header_bytes=1024)
    assert read_gguf_metadata(path, max_header_bytes=len(good) + 1)["general.name"] == "Qwen3.8 27B"
    with pytest.raises(GGUFError):
        read_gguf_metadata(path, max_header_bytes=8)
    with pytest.raises(GGUFError, match="regular file"):
        read_gguf_metadata(tmp_path)
    # Implausible counts, unknown types, duplicate keys and bad UTF-8 are all refused.
    with pytest.raises(GGUFError, match="metadata count"):
        parse_gguf_header(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1 << 40))
    unknown_type = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1) + _string("k") + struct.pack("<I", 99)
    with pytest.raises(GGUFError, match="unknown gguf value type"):
        parse_gguf_header(unknown_type)
    with pytest.raises(GGUFError, match="duplicate"):
        parse_gguf_header(build_gguf([("general.name", "str", "a"), ("general.name", "str", "b")]))
    bad_utf8 = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1) + _string("general.name") + struct.pack(
        "<I", 8) + struct.pack("<Q", 2) + b"\xff\xfe"
    with pytest.raises(GGUFError, match="UTF-8"):
        parse_gguf_header(bad_utf8)
    with pytest.raises(GGUFError, match="general.architecture"):
        gguf_summary(_write(tmp_path / "noarch.gguf", build_gguf([("general.name", "str", "x")])))
    with pytest.raises(GGUFError):
        template_hash("")


def test_template_hash_matches_props_chat_template_sha_from_pilot_fixture(tmp_path):
    template = pilot_template()
    assert len(template) == 8952
    path = _write(tmp_path / "pilot-Q4_K_M.gguf", build_gguf(fixture_entries(template)))
    assert template_hash(read_gguf_metadata(path)["tokenizer.chat_template"]) == PILOT_TEMPLATE_ID
    assert gguf_summary(path)["template_hash"] == PILOT_TEMPLATE_ID
    # The evaluator labels retrieval samples with sha256(props.chat_template); the same digest must match.
    assert "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest() == PILOT_TEMPLATE_ID


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _nested_array_header(depth: int) -> bytes:
    nested = struct.pack("<I", 4) + struct.pack("<Q", 0)  # innermost: an empty u32 array
    for _ in range(depth):  # each level: a one-element array whose element type is itself an array
        nested = struct.pack("<I", 9) + struct.pack("<Q", 1) + nested
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 2) + _string("deep") + struct.pack("<I", 9) + nested
    return header + _string("general.name") + _value("str", "x")


def test_deeply_nested_arrays_fail_closed_with_gguf_error():
    """REV-C1-03: 5000 (RecursionError today) and 64 nested array types raise GGUFError; shallow nesting skips."""
    from llmbench.containers.gguf import MAX_ARRAY_DEPTH
    for depth in (5000, 64):
        with pytest.raises(GGUFError, match="nest"):
            parse_gguf_header(_nested_array_header(depth), keys=None)
    assert 1 <= MAX_ARRAY_DEPTH <= 8
    shallow = build_gguf([("tokenizer.ggml.nested", "arr-arr-u32", [[1, 2], []]), ("general.name", "str", "ok")])
    assert parse_gguf_header(shallow, keys=None)["general.name"] == "ok"


def _tensor_infos(types):
    """A GGUF tensor-info block: name, n_dims, dims, ggml type, data offset - one entry per requested type."""
    return b"".join(_string(f"blk.{index}.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 8, 8)
                    + struct.pack("<I", code) + struct.pack("<Q", index * 64) for index, code in enumerate(types))


def test_weights_in_a_format_the_header_cannot_name_are_labelled_from_the_tensors(tmp_path):
    """NVFP4 GGUFs declare general.file_type 7 (Q8_0) because llama_ftype has no NVFP4 value. Two different models
    then both read as "Q8_0"; the tensor types are what the weights really are."""
    entries = [item if item[0] != "general.file_type" else ("general.file_type", "u32", 7) for item in fixture_entries()]
    nvfp4 = [0] * 20 + [40] * 12 + [8] * 3                                 # F32 norms, FP4 weights, a few Q8_0
    summary = gguf_summary(_write(tmp_path / "nvfp4.gguf", build_gguf(entries, tensors=len(nvfp4)) + _tensor_infos(nvfp4)))
    assert summary["declared_quantization"] == "Q8_0" and summary["weight_type"] == "NVFP4"
    assert summary["quantization"] == "NVFP4"
    assert summary["tensor_types"] == {"F32": 20, "NVFP4": 12, "Q8_0": 3}

    ordinary = [0] * 20 + [12] * 12 + [14] * 3                             # a normal Q4_K_M mix keeps its header name
    entries[[item[0] for item in entries].index("general.file_type")] = ("general.file_type", "u32", 15)
    summary = gguf_summary(_write(tmp_path / "q4.gguf", build_gguf(entries, tensors=len(ordinary)) + _tensor_infos(ordinary)))
    assert (summary["quantization"], summary["weight_type"]) == ("Q4_K_M", "Q4_K")

    unknown = [0] * 2 + [77] * 5                                           # a type this table does not know
    summary = gguf_summary(_write(tmp_path / "new.gguf", build_gguf(entries, tensors=len(unknown)) + _tensor_infos(unknown)))
    assert summary["weight_type"] == "type_77" and summary["quantization"] == "Q4_K_M"


def test_a_missing_or_truncated_tensor_block_is_unknown_not_guessed(tmp_path):
    declared_but_absent = gguf_summary(_write(tmp_path / "a.gguf", build_gguf(fixture_entries(), tensors=5)))
    assert declared_but_absent["tensor_types"] is None and declared_but_absent["weight_type"] is None
    assert declared_but_absent["quantization"] == "Q4_K_M"                 # the header still answers
    truncated = build_gguf(fixture_entries(), tensors=3) + _tensor_infos([40, 40, 40])[:-10]
    assert gguf_summary(_write(tmp_path / "b.gguf", truncated))["tensor_types"] is None
    assert gguf_summary(_write(tmp_path / "c.gguf", build_gguf(fixture_entries(), tensors=0)))["tensor_types"] is None

"""Pure GGUF v2/v3 header reader: scalar metadata only, arrays skipped, bounded read, no model load.

The header is read into memory once (at most ``max_header_bytes``) and parsed from that buffer.
Anything malformed, truncated or outside the bound raises ``GGUFError`` instead of guessing.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = frozenset({2, 3})
DEFAULT_MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_KV_COUNT = 65536
MAX_KEY_BYTES = 65536
MAX_STRING_BYTES = 16 * 1024 * 1024
MAX_ARRAY_LENGTH = 1 << 32
MAX_ARRAY_DEPTH = 4  # real GGUF headers nest arrays at most one level; deeper is malformed, never recursed into
DEFAULT_KEYS = ("tokenizer.chat_template", "general.architecture", "general.name")

# gguf value types (ggml/gguf.h): code -> struct format for scalars.
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9

# llama.cpp ``general.file_type`` enum (llama_ftype) -> quantization name.
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
    12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
}


# ggml tensor types (ggml.h ``enum ggml_type``). A code missing here is reported as ``type_<n>``, never guessed.
TENSOR_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
}
UNQUANTIZED_TENSOR_TYPES = frozenset({"F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"})
TENSOR_ONLY_QUANTIZATIONS = frozenset({"NVFP4"})
"""Weight formats ``llama_ftype`` has no value for. A converter writing one must put SOMETHING in
``general.file_type`` (NVFP4 files in the wild say 7, i.e. Q8_0, after the handful of Q8_0 tensors beside the FP4
ones), so for these the tensors, not the header, name the quantization."""
MAX_TENSOR_COUNT = 1 << 20
MAX_TENSOR_DIMS = 8


class GGUFError(ValueError):
    """The file is not a GGUF v2/v3 header this reader can trust."""


class _Cursor:
    def __init__(self, data: bytes, *, bounded: bool) -> None:
        self.data, self.offset, self.bounded = data, 0, bounded

    def take(self, count: int) -> bytes:
        if count < 0:
            raise GGUFError("negative length in header")
        end = self.offset + count
        if end > len(self.data):
            raise GGUFError("header truncated" if not self.bounded else
                            "header exceeds the bounded read (truncated file or header larger than the bound)")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def skip(self, count: int) -> None:
        self.take(count)

    def scalar(self, code: int) -> Any:
        fmt = _SCALAR[code]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def u32(self) -> int:
        return self.scalar(4)

    def u64(self) -> int:
        return self.scalar(10)

    def string(self, *, limit: int = MAX_STRING_BYTES) -> str:
        length = self.u64()
        if length > limit:
            raise GGUFError(f"string of {length} bytes exceeds the {limit} byte limit")
        try:
            return self.take(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("string value is not valid UTF-8") from exc

    def skip_value(self, code: int, *, depth: int = 0) -> None:
        if code in _SCALAR:
            self.skip(struct.calcsize(_SCALAR[code]))
        elif code == _STRING:
            self.skip(self._checked_length(self.u64(), MAX_STRING_BYTES))
        elif code == _ARRAY:
            self.skip_array(depth=depth + 1)
        else:
            raise GGUFError(f"unknown gguf value type {code}")

    def skip_array(self, *, depth: int = 1) -> None:
        if depth > MAX_ARRAY_DEPTH:
            raise GGUFError(f"array nests deeper than {MAX_ARRAY_DEPTH} levels")
        element = self.u32()
        length = self._checked_length(self.u64(), MAX_ARRAY_LENGTH)
        if element in _SCALAR:
            self.skip(length * struct.calcsize(_SCALAR[element]))
        elif element in {_STRING, _ARRAY}:
            for _ in range(length):
                self.skip_value(element, depth=depth)
        else:
            raise GGUFError(f"unknown gguf array element type {element}")

    @staticmethod
    def _checked_length(value: int, limit: int) -> int:
        if value > limit:
            raise GGUFError(f"length {value} exceeds the limit {limit}")
        return value


def parse_gguf_header(data: bytes, *, keys: tuple[str, ...] | None = DEFAULT_KEYS, bounded: bool = False) -> dict:
    """Parse scalar key/values from GGUF bytes; arrays are skipped, never materialised.

    ``keys=None`` returns every scalar entry. The result always carries ``version``, ``tensor_count`` and
    ``kv_count`` under the reserved ``__gguf__`` key.
    """
    cursor = _Cursor(bytes(data), bounded=bounded)
    if cursor.take(4) != MAGIC:
        raise GGUFError("not a GGUF file (bad magic)")
    version = cursor.u32()
    if version not in SUPPORTED_VERSIONS:
        raise GGUFError(f"unsupported GGUF version {version}; only v2 and v3 headers are read")
    tensor_count, kv_count = cursor.u64(), cursor.u64()
    if kv_count > MAX_KV_COUNT:
        raise GGUFError(f"implausible metadata count {kv_count}")
    wanted = None if keys is None else set(keys)
    found: dict[str, Any] = {}
    for _ in range(kv_count):
        key = cursor.string(limit=MAX_KEY_BYTES)
        if not key:
            raise GGUFError("empty metadata key")
        code = cursor.u32()
        if code == _ARRAY:
            cursor.skip_array()
            continue
        if code not in _SCALAR and code != _STRING:
            raise GGUFError(f"unknown gguf value type {code} for {key}")
        if wanted is not None and key not in wanted:
            cursor.skip_value(code)
            continue
        if key in found:
            raise GGUFError(f"duplicate metadata key {key}")
        found[key] = cursor.string() if code == _STRING else cursor.scalar(code)
        if wanted is not None and wanted <= found.keys():
            break
    found["__gguf__"] = {"version": version, "tensor_count": tensor_count, "kv_count": kv_count,
                         "tensor_types": _tensor_type_counts(cursor, tensor_count) if wanted is None else None}
    return found


def _tensor_type_counts(cursor: _Cursor, tensor_count: int) -> dict[str, int] | None:
    """How many tensors use each ggml type, read from the tensor-info block that follows the metadata.

    ``None`` when the block is absent, truncated or implausible: the caller then knows the weights were not
    inspected, which is different from knowing what they are.
    """
    if not 0 < tensor_count <= MAX_TENSOR_COUNT:
        return None
    counts: dict[str, int] = {}
    try:
        for _ in range(tensor_count):
            cursor.string(limit=MAX_KEY_BYTES)
            dims = cursor.u32()
            if dims > MAX_TENSOR_DIMS:
                return None
            cursor.skip(8 * dims)
            code = cursor.u32()
            cursor.skip(8)  # data offset
            name = TENSOR_TYPES.get(code, f"type_{code}")
            counts[name] = counts.get(name, 0) + 1
    except GGUFError:
        return None
    return counts


def weight_type_of(tensor_types: dict[str, int] | None) -> str | None:
    """The type most quantized tensors use (ties: the name that sorts first). ``None`` when nothing is quantized."""
    quantized = {name: count for name, count in (tensor_types or {}).items() if name not in UNQUANTIZED_TENSOR_TYPES}
    if not quantized:
        return None
    return min(quantized, key=lambda name: (-quantized[name], name))


def read_gguf_metadata(path: str | Path, *, keys: tuple[str, ...] | None = DEFAULT_KEYS,
                       max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES) -> dict:
    """Read at most ``max_header_bytes`` from ``path`` and parse the scalar metadata it holds."""
    if type(max_header_bytes) is not int or max_header_bytes < 32:
        raise GGUFError("max_header_bytes must be an integer of at least 32")
    target = Path(path)
    if not target.is_file():
        raise GGUFError(f"not a regular file: {target}")
    with target.open("rb") as handle:
        data = handle.read(max_header_bytes)
    return parse_gguf_header(data, keys=keys, bounded=len(data) >= max_header_bytes)


def template_hash(chat_template: str) -> str:
    """Identity of a serving template, byte-for-byte as ``/props`` reports it: ``sha256:<hex>``."""
    if type(chat_template) is not str or not chat_template:
        raise GGUFError("chat template must be a nonempty string")
    return "sha256:" + hashlib.sha256(chat_template.encode("utf-8")).hexdigest()


def gguf_summary(path: str | Path, *, max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES) -> dict:
    """Architecture-resolved facts the session derivation needs; absent facts are ``None``, never guessed."""
    raw = read_gguf_metadata(path, keys=None, max_header_bytes=max_header_bytes)
    architecture = raw.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise GGUFError("general.architecture is missing")

    def arch(name: str) -> Any:
        return raw.get(f"{architecture}.{name}")

    template = raw.get("tokenizer.chat_template")
    file_type = raw.get("general.file_type")
    declared = FILE_TYPES.get(file_type) if type(file_type) is int else None
    tensor_types = raw["__gguf__"].get("tensor_types")
    weights = weight_type_of(tensor_types)
    # The header names the quantization unless the weights are in a format the header has no value for.
    quantization = weights if weights in TENSOR_ONLY_QUANTIZATIONS else declared
    return {"architecture": architecture, "name": raw.get("general.name"),
            "chat_template": template if isinstance(template, str) and template else None,
            "template_hash": template_hash(template) if isinstance(template, str) and template else None,
            "n_ctx_train": arch("context_length") if type(arch("context_length")) is int else None,
            "block_count": arch("block_count") if type(arch("block_count")) is int else None,
            "nextn_predict_layers": arch("nextn_predict_layers")
            if type(arch("nextn_predict_layers")) is int else None,
            "file_type": file_type if type(file_type) is int else None,
            "quantization": quantization, "declared_quantization": declared,
            "weight_type": weights, "tensor_types": tensor_types,
            "gguf": raw["__gguf__"]}

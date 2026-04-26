"""Compressed precompute file format: msgpack + u8 quantization + zstd.

~5x smaller than gzipped JSON while preserving decision-level correctness
(strategy probabilities quantized to 1/255, which is below CFR+ convergence error).

File extension: .mpk.zst
Schema (short keys chosen for msgpack compactness):
  fl (str)        flop_label like "3c7hTh"
  k  (u8)         number of buckets (16)
  it (u32)        iteration count
  sb (u32)        stack_bb
  hv (f32)        hero_value
  oc (bytes)      oop_bucket_of_combo, 1326 bytes (i8 as u8 two's-complement)
  ic (bytes)      ip_bucket_of_combo, 1326 bytes
  be (bytes)      bucket_ehs_range u8-quantized, k*2 bytes
  ve (bytes)      villain_bucket_ehs_range, k*2 bytes
  bh (list[list[str]])  bucket_hands
  vh (list[list[str]])  villain_bucket_hands
  al (list[str])        root action_labels
  rs (bytes)            root_strategy u8, k * len(al) bytes
  n  (list[node])       decision nodes, each:
     i (u32) node_id
     p (i8)  player
     s (str) street "flop"|"turn"
     pa (str) path
     tc (str|None) turn_card
     al (list[str]) action_labels
     q  (bytes) strategy u8, k * len(al) bytes

Dequantization on read: prob_float = byte / 255.0, then row-normalize so
each bucket's strategy sums to 1.0 exactly.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import msgpack
import zstandard as zstd

Q_MAX = 255
_ZSTD_LEVEL = 19
_DECOMPRESSOR = zstd.ZstdDecompressor()
_COMPRESSOR = zstd.ZstdCompressor(level=_ZSTD_LEVEL)


def _q_prob(p: float) -> int:
    return max(0, min(Q_MAX, round(p * Q_MAX)))


def _q_strategy_2d(strat: List[List[float]]) -> bytes:
    out = bytearray()
    for row in strat:
        for p in row:
            out.append(_q_prob(p))
    return bytes(out)


def _q_ehs_ranges(ranges: List[List[float]]) -> bytes:
    out = bytearray()
    for r in ranges:
        for v in r:
            out.append(_q_prob(v))
    return bytes(out)


def _dq_strategy_2d(packed: bytes, k: int, na: int) -> List[List[float]]:
    """Dequantize and re-normalize strategy so each bucket row sums to 1."""
    out = []
    for b in range(k):
        off = b * na
        row = [packed[off + a] / Q_MAX for a in range(na)]
        s = sum(row)
        if s > 0:
            row = [x / s for x in row]
        out.append(row)
    return out


def _dq_ehs_ranges(packed: bytes, k: int) -> List[List[float]]:
    out = []
    for b in range(k):
        lo = packed[b * 2] / Q_MAX
        hi = packed[b * 2 + 1] / Q_MAX
        out.append([lo, hi])
    return out


def encode(data: Dict[str, Any]) -> bytes:
    """Legacy JSON dict → compressed msgpack bytes."""
    k = data["k_buckets"]
    nodes_out = []
    for n in data["nodes"]:
        nodes_out.append({
            "i":  n["node_id"],
            "p":  n["player"],
            "s":  n["street"],
            "pa": n["path"],
            "tc": n.get("turn_card"),
            "al": n["action_labels"],
            "q":  _q_strategy_2d(n["strategy"]),
        })
    compact = {
        "fl": data["flop_label"],
        "k":  k,
        "it": data.get("iterations", 0),
        "sb": data.get("stack_bb", 0),
        "hv": float(data.get("hero_value", 0.0)),
        "oc": bytes(x & 0xFF for x in data["oop_bucket_of_combo"]),
        "ic": bytes(x & 0xFF for x in data["ip_bucket_of_combo"]),
        "be": _q_ehs_ranges(data.get("bucket_ehs_range", [[0, 0]] * k)),
        "ve": _q_ehs_ranges(data.get("villain_bucket_ehs_range", [[0, 0]] * k)),
        "bh": data.get("bucket_hands", []),
        "vh": data.get("villain_bucket_hands", []),
        "al": data["action_labels"],
        "rs": _q_strategy_2d(data["root_strategy"]),
        "n":  nodes_out,
    }
    return _COMPRESSOR.compress(msgpack.packb(compact, use_bin_type=True))


def decode(blob: bytes) -> Dict[str, Any]:
    """Compressed msgpack bytes → legacy JSON-shape dict."""
    raw = _DECOMPRESSOR.decompress(blob)
    c = msgpack.unpackb(raw, raw=False)
    k = c["k"]
    oc = list(bytes(c["oc"]))  # convert to int list
    ic = list(bytes(c["ic"]))
    # back to i8 (values above 127 were -128..-1)
    oc = [(v - 256) if v > 127 else v for v in oc]
    ic = [(v - 256) if v > 127 else v for v in ic]

    root_al = c["al"]
    nodes_out = []
    for n in c["n"]:
        al = n["al"]
        nodes_out.append({
            "node_id":       n["i"],
            "player":        n["p"],
            "street":        n["s"],
            "path":          n["pa"],
            "turn_card":     n["tc"],
            "action_labels": al,
            "strategy":      _dq_strategy_2d(n["q"], k, len(al)),
        })
    return {
        "flop_label":                c["fl"],
        "k_buckets":                 k,
        "iterations":                c["it"],
        "stack_bb":                  c["sb"],
        "hero_value":                c["hv"],
        "oop_bucket_of_combo":       oc,
        "ip_bucket_of_combo":        ic,
        "bucket_ehs_range":          _dq_ehs_ranges(c["be"], k),
        "villain_bucket_ehs_range":  _dq_ehs_ranges(c["ve"], k),
        "bucket_hands":              c.get("bh", []),
        "villain_bucket_hands":      c.get("vh", []),
        "action_labels":             root_al,
        "root_strategy":             _dq_strategy_2d(c["rs"], k, len(root_al)),
        "nodes":                     nodes_out,
    }


def load_any(path: Path) -> Dict[str, Any]:
    """Load a precompute file regardless of format (.json, .json.gz, .mpk.zst)."""
    name = path.name
    if name.endswith(".mpk.zst"):
        return decode(path.read_bytes())
    if name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    if name.endswith(".json"):
        return json.loads(path.read_text())
    raise ValueError(f"Unknown precompute file extension: {path.name}")


def convert_file(src: Path, dst: Path) -> int:
    """Read any-format precompute file, write .mpk.zst. Returns output size in bytes."""
    data = load_any(src)
    blob = encode(data)
    dst.write_bytes(blob)
    return len(blob)

"""Solver wrapper for the /poker skill.

Two layers:
1. `lookup_cached(spec)` — try precomputed table; return solution or None.
2. `solve_on_demand(spec)` — invoke Rust `solve` binary as subprocess.

Top-level `solve(spec)` does (1) then (2) automatically.

`spec` schema matches `src/bin/solve.rs`:
{
  "game": "river" | "turn" | "flop" | "preflop",
  "board": [...],
  "pot": float,
  "stacks": [float, float],
  "first_to_act": 0 | 1,
  "hero_range": str,
  "villain_range": str,
  "iterations": int,
  ...
}

Returns a dict with at minimum {"hero_value", "strategy"}, plus optional
{"all_strategies", "exploitability"} when requested.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def project_root() -> Path:
    """Locate the project root by walking up from this file."""
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "Cargo.toml").exists() and (p / "python").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not locate poker-gto-engine project root")


def find_solve_binary() -> Optional[Path]:
    root = project_root()
    for cand in [root / "target" / "release" / "solve", root / "target" / "debug" / "solve"]:
        if cand.exists() and os.access(cand, os.X_OK):
            return cand
    return None


def precompute_dir() -> Path:
    return project_root() / "precompute_out"


def spec_key(spec: Dict[str, Any]) -> str:
    """Stable hash of the solver-relevant fields. Order-insensitive on dict
    iteration via sort_keys."""
    canonical = json.dumps({
        k: spec[k] for k in sorted(spec)
        if k in {"game", "board", "pot", "stacks", "first_to_act",
                 "hero_range", "villain_range", "max_raises", "turn_max_raises",
                 "river_max_raises"}
    })
    return hashlib.sha1(canonical.encode()).hexdigest()[:16]


def lookup_cached(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Search for a matching precomputed solution.

    Two layers searched in order:
    1. by_spec/  — exact-spec hash lookup (populated by the `solve` binary
       on every run). Hits when the user re-asks the same spot.
    2. precompute_out/<job>/ — bucketed precompute output keyed by
       (game, board_label, stack_bb). Looser match — if the spot's flop
       texture and stack depth are in the cache, we use it.

    On miss, caller should fall through to on-demand solve.
    """
    cache_dir = precompute_dir()
    if not cache_dir.exists():
        return None

    # Layer 1: exact-spec cache (managed by the Rust solve binary)
    key = spec_key(spec)
    by_spec = cache_dir / "by_spec" / key[:2] / f"{key}.json"
    if by_spec.exists():
        return json.loads(by_spec.read_text())

    # Layer 2: precomputed flop solutions (bucketed)
    if spec.get("game") == "flop" and "board" in spec:
        board_label = "".join(spec["board"])
        stack_bb = (spec.get("stacks", [200.0, 200.0])[0]) // 1
        # Search any precompute job dir
        for job_dir in cache_dir.iterdir():
            if not job_dir.is_dir():
                continue
            cand = job_dir / f"flop_{board_label}.json"
            if cand.exists():
                data = json.loads(cand.read_text())
                # Only accept if stack matches (within 5 BB)
                if abs(data.get("stack_bb", 0) - stack_bb) <= 5:
                    return _adapt_bucketed_to_solve_output(data, spec)
    return None


def _adapt_bucketed_to_solve_output(bucketed: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a bucketed precompute solution file into the same shape that
    the `solve` binary's on-demand output uses, so downstream formatters work
    with either source uniformly."""
    return {
        "game": spec.get("game", "flop"),
        "board": spec.get("board", []),
        "pot": spec.get("pot", 6.0),
        "stacks": spec.get("stacks", [bucketed.get("stack_bb", 200.0)] * 2),
        "first_to_act": spec.get("first_to_act", 0),
        "iterations": bucketed.get("iterations", 0),
        "hero_value": bucketed.get("hero_value", 0.0),
        # Bucketed solutions store strategy per-bucket, not per-combo.
        # Caller needs to know which bucket their hero combo falls into.
        # For now we return the raw bucket strategies; the format_table layer
        # handles the bucket→combo lookup.
        "_bucketed": True,
        "k_buckets": bucketed.get("k_buckets"),
        "bucket_root_strategy": bucketed.get("root_strategy"),
        "action_labels": bucketed.get("action_labels"),
        "_source_file": str(bucketed.get("flop_label", "")),
    }


def solve_on_demand(spec: Dict[str, Any], timeout_sec: float = 1800.0) -> Dict[str, Any]:
    binary = find_solve_binary()
    if binary is None:
        raise RuntimeError(
            "Rust solver binary not built. Run: "
            "cd ~/projects/poker-gto-engine && cargo build --release --bin solve"
        )
    spec_json = json.dumps(spec)
    proc = subprocess.run(
        [str(binary)],
        input=spec_json, capture_output=True, text=True,
        timeout=timeout_sec, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Rust solve failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Solver output not JSON: {e}\nRaw: {proc.stdout[:500]}")


def solve(spec: Dict[str, Any], force_on_demand: bool = False) -> Dict[str, Any]:
    """Try cache, fall back to on-demand. Returns solver output dict."""
    if not force_on_demand:
        cached = lookup_cached(spec)
        if cached is not None:
            cached["_source"] = "cached"
            return cached
    result = solve_on_demand(spec)
    result["_source"] = "on_demand"
    return result


def cache_status() -> Dict[str, Any]:
    """Report what's available in the precompute cache. For diagnostics."""
    cache_dir = precompute_dir()
    out: Dict[str, Any] = {"cache_dir": str(cache_dir), "exists": cache_dir.exists()}
    if cache_dir.exists():
        files = list(cache_dir.rglob("*.json"))
        out["n_files"] = len(files)
        out["files"] = sorted(str(f.relative_to(cache_dir)) for f in files[:20])
        if len(files) > 20:
            out["files"].append(f"... and {len(files) - 20} more")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(json.dumps(cache_status(), indent=2))
    else:
        # Smoke test
        spec = {
            "game": "river",
            "board": ["Ad", "Kh", "7s", "3c", "2d"],
            "pot": 100.0,
            "stacks": [200.0, 200.0],
            "first_to_act": 0,
            "hero_range": "AhAc",
            "villain_range": "2s2h",
            "iterations": 100,
        }
        result = solve(spec)
        print(f"source: {result.get('_source')}")
        print(f"hero_value: {result['hero_value']:.3f}")

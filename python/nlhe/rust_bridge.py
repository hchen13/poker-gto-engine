"""Wrapper that calls the Rust `solve` binary as a subprocess.

The Python solvers in this package are reference implementations and good for
correctness, but the Rust port (in `target/release/solve`) is 10-75× faster
on the same workloads. This bridge lets the existing Python CLI / skill use
Rust transparently when the binary is built.

Usage:

    from python.nlhe.rust_bridge import solve_via_rust
    result = solve_via_rust({
        "game": "river",
        "board": ["Ad","Kh","7s","3c","2d"],
        "pot": 100, "stacks": [200, 200],
        "first_to_act": 0,
        "hero_range": "AhAc",
        "villain_range": "2s2h",
        "iterations": 500,
    })

Returns the JSON output of the Rust binary as a Python dict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class RustBridgeError(RuntimeError):
    pass


def find_solve_binary() -> Optional[Path]:
    """Locate the compiled `solve` binary. Returns None if not built."""
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "target" / "release" / "solve",
        project_root / "target" / "debug" / "solve",
    ]
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return path
    return None


def solve_via_rust(spec: Dict[str, Any], timeout_sec: float = 600.0) -> Dict[str, Any]:
    """Invoke the Rust solver binary. Spec schema matches `src/bin/solve.rs`."""
    binary = find_solve_binary()
    if binary is None:
        raise RustBridgeError(
            "Rust solver binary not found. Build with: cargo build --release --bin solve"
        )
    spec_json = json.dumps(spec)
    try:
        proc = subprocess.run(
            [str(binary)],
            input=spec_json,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RustBridgeError(f"Rust solver timed out after {timeout_sec}s")
    if proc.returncode != 0:
        raise RustBridgeError(
            f"Rust solver failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RustBridgeError(f"Rust solver output not JSON: {e}\n{proc.stdout[:500]}")


def is_rust_available() -> bool:
    return find_solve_binary() is not None


if __name__ == "__main__":
    # Smoke test
    spec = {
        "game": "river",
        "board": ["Ad", "Kh", "7s", "3c", "2d"],
        "pot": 100,
        "stacks": [200, 200],
        "first_to_act": 0,
        "hero_range": "AhAc",
        "villain_range": "2s2h",
        "iterations": 100,
    }
    if not is_rust_available():
        print("Rust solver not built. Run: cargo build --release --bin solve")
        sys.exit(1)
    result = solve_via_rust(spec)
    print(f"hero_value: {result['hero_value']:.3f}")
    print(f"strategies: {len(result['strategy'])} hero combos")

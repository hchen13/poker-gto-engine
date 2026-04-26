"""Regression tests for skill/poker/bin/ CLI tools.

Covers: find_flop.py, paths.py, query.py (in-range + approximated fallback),
solve_river.py (OOP + IP + out-of-range fallback).

Each test invokes the CLI as a subprocess to exercise the user-facing path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIN = PROJECT_ROOT / "skill" / "poker" / "bin"


def run_cli(script: str, *args: str, timeout: float = 60.0) -> dict:
    """Run a CLI script and return parsed JSON from stdout. Raises on non-zero exit."""
    proc = subprocess.run(
        [sys.executable, str(BIN / script), *args],
        capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT_ROOT),
    )
    if proc.returncode != 0:
        pytest.fail(f"{script} exited {proc.returncode}\nstderr:\n{proc.stderr}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"{script} output not JSON: {e}\nstdout:\n{proc.stdout[:500]}")


def run_cli_expecting_error(script: str, *args: str, timeout: float = 60.0) -> dict:
    """Run a CLI script expecting non-zero exit. Tools vary in where they emit
    error JSON (stderr or stdout), so try both."""
    proc = subprocess.run(
        [sys.executable, str(BIN / script), *args],
        capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode != 0, f"expected error, got exit 0.\nstdout:\n{proc.stdout}"
    for stream in (proc.stderr, proc.stdout):
        stream = stream.strip()
        if stream:
            try:
                return json.loads(stream)
            except json.JSONDecodeError:
                continue
    pytest.fail(f"{script} error output not JSON.\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")


# ---------- find_flop.py ----------


class TestFindFlop:
    def test_rank_texture_rainbow(self):
        r = run_cli("find_flop.py", "3bet_called", "500", "AK7r")
        assert r["texture"] == "rainbow"
        assert len(r["flop_label"]) == 6
        # all 3 cards different suits
        suits = {r["flop_label"][1], r["flop_label"][3], r["flop_label"][5]}
        assert len(suits) == 3

    def test_rank_texture_flush_draw(self):
        r = run_cli("find_flop.py", "3bet_called", "500", "AK7s")
        assert r["texture"] == "flush_draw"

    def test_exact_cards_match(self):
        # Provide exact 6-char label — should resolve to the same or a rainbow equivalent
        r = run_cli("find_flop.py", "3bet_called", "500", "7cKhAd")
        assert r["flop_label"] == "7cKhAd"

    def test_invalid_action_line_errors(self):
        r = run_cli_expecting_error("find_flop.py", "bogus_line", "200", "AK7r")
        assert "error" in r


# ---------- paths.py ----------


class TestPaths:
    def test_has_flop_and_turn_paths(self):
        r = run_cli("paths.py", "3bet_called", "500", "AK7r")
        assert r["oop_flop_paths"], "OOP flop paths should be non-empty"
        assert r["ip_flop_paths"], "IP flop paths should be non-empty"
        assert r["turn_count"] > 0
        # OOP acts first at root → empty path should appear in OOP list
        assert "" in r["oop_flop_paths"]

    def test_all_variants_queryable(self):
        """All 7 action_lines should produce paths output without error."""
        for al in ["limped", "sr_called", "sr_called_ip_caller",
                   "3bet_called", "3bet_called_ip3bet",
                   "4bet_called", "4bet_called_ip_caller"]:
            r = run_cli("paths.py", al, "200", "K62r")
            assert r["total_nodes"] > 0, f"{al}: no nodes"


# ---------- query.py ----------


class TestQueryInRange:
    def test_oop_3bettor_at_root(self):
        r = run_cli("query.py", "3bet_called", "500", "AK7r", "AKo", "oop", "")
        assert r["bucket_idx"] >= 0
        assert "AKo" in r["bucket_hands"]
        # sum of strategy probs should be ~1
        total = sum(r["strategy"])
        assert 0.99 < total < 1.01

    def test_ip_caller_after_check(self):
        r = run_cli("query.py", "3bet_called", "500", "AK7r", "KQo", "ip", "check")
        assert r["bucket_idx"] >= 0
        assert "KQo" in r["bucket_hands"]

    def test_turn_node(self):
        r = run_cli("query.py", "3bet_called", "500", "AK7r", "AKo", "oop",
                    "check/check", "--turn", "Ts")
        assert r["bucket_idx"] >= 0
        assert r["node_street"] == "turn"


class TestQueryApproximated:
    def test_kqo_as_ip_in_sr_called_ip_caller(self):
        """KQo isn't in the HU-calibrated IP caller range — should approximate by EHS."""
        r = run_cli("query.py", "sr_called_ip_caller", "200", "K62r", "KQo", "ip", "check")
        assert r["bucket_idx"] >= 0
        # Approximation marker in bucket_hands first entry
        assert any("~KQo" in h for h in r["bucket_hands"]), \
            f"expected approximation marker, got {r['bucket_hands'][:3]}"

    def test_error_still_raised_when_possible(self):
        """Totally invalid hand input should still produce a clear error."""
        r = run_cli_expecting_error("query.py", "3bet_called", "500", "AK7r",
                                     "BOGUS", "oop", "")
        assert "error" in r


# ---------- solve_river.py ----------


class TestIsoKey:
    """Canonical iso-class key consistency between Python and Rust."""

    def test_suit_permutation_invariance(self):
        import sys as _sys
        _sys.path.insert(0, str(PROJECT_ROOT))
        from skill.poker.lib.query_precompute import flop_iso_key
        # Same iso class: AKs 3-flush-draw pattern vs. suit-permuted variant
        k1 = flop_iso_key(["As", "Kh", "7h"])
        k2 = flop_iso_key(["Ac", "Kd", "7d"])
        assert k1 == k2
        # Different class (rainbow instead of 2-tone)
        k3 = flop_iso_key(["As", "Kh", "7d"])
        assert k1 != k3

    def test_all_22100_flops_map_to_1755_classes(self):
        import sys as _sys
        _sys.path.insert(0, str(PROJECT_ROOT))
        from skill.poker.lib.query_precompute import flop_iso_key
        from python.nlhe.cards import INDEX_TO_CARD
        seen = set()
        count = 0
        for a in range(52):
            for b in range(a + 1, 52):
                for c in range(b + 1, 52):
                    cards = [INDEX_TO_CARD[a], INDEX_TO_CARD[b], INDEX_TO_CARD[c]]
                    seen.add(flop_iso_key(cards))
                    count += 1
        assert count == 22100
        assert len(seen) == 1755


class TestSubgameOnDemand:
    """On-demand turn/flop solves with manual ranges."""

    # Compact ranges that expand to real HU-ish sets via parse_range
    OOP_RANGE = ("22-JJ, A2s, A5s, ATs, K2s, K5s, K9s, KTs, Q2s, Q5s, Q9s, "
                 "J3s, J7s, J9s, JTs, T5s, T8s, T9s, 98s, 87s, 76s, 65s, 54s, "
                 "AJo, ATo, KJo, KTo, QJo, QTo, JTo, T9o, 98o, 87o")
    IP_RANGE = ("22+, A2s, A5s, ATs, AJs, AQs, AKs, K2s, K5s, K9s, KTs, KJs, "
                "KQs, Q5s, Q9s, QTs, QJs, J5s, J7s, J9s, JTs, T5s, T8s, T9s, "
                "98s, 87s, 76s, 65s, 54s, A2o, A5o, ATo, AJo, AQo, AKo, KTo, "
                "KJo, KQo, QTo, QJo, JTo, T9o")

    def test_turn_manual_runs_under_10s(self):
        import time
        t0 = time.time()
        r = run_cli("solve_turn_manual.py",
                    "--board", "Tc 7h 3d 9c",
                    "--oop-range", self.OOP_RANGE,
                    "--ip-range", self.IP_RANGE,
                    "--pot", "12", "--oop-stack", "194", "--ip-stack", "194",
                    "--hand", "JTs", "--position", "oop",
                    "--iterations", "100",
                    timeout=60)
        assert time.time() - t0 < 10, "turn on-demand should be fast"
        assert r["street"] == "turn"
        assert r["hero_bucket_idx"] >= 0
        assert len(r["action_labels"]) > 0

    def test_flop_manual_runs_under_2min(self):
        import time
        t0 = time.time()
        r = run_cli("solve_flop_manual.py",
                    "--board", "Tc 7h 3d",
                    "--oop-range", self.OOP_RANGE,
                    "--ip-range", self.IP_RANGE,
                    "--pot", "12", "--oop-stack", "194", "--ip-stack", "194",
                    "--hand", "JTs", "--position", "oop",
                    "--iterations", "100",
                    timeout=150)
        assert time.time() - t0 < 120, "flop on-demand should fit 2-min budget"
        assert r["street"] in ("flop", "turn")
        assert r["hero_bucket_idx"] >= 0

    def test_turn_ip_responses_include_multiple_actions(self):
        r = run_cli("solve_turn_manual.py",
                    "--board", "Tc 7h 3d 9c",
                    "--oop-range", self.OOP_RANGE,
                    "--ip-range", self.IP_RANGE,
                    "--pot", "12", "--oop-stack", "194", "--ip-stack", "194",
                    "--hand", "JTs", "--position", "ip",
                    "--iterations", "100",
                    timeout=60)
        assert r["street"] == "turn"
        # IP hero → ip_responses should have entries for OOP's plausible actions
        assert isinstance(r["ip_responses"], dict)
        assert len(r["ip_responses"]) >= 2


class TestSolveRiver:
    def test_oop_river_in_range(self):
        r = run_cli("solve_river.py", "3bet_called", "500", "AK7r",
                    "check/check/check/check",
                    "--turn", "Ts", "--river", "2h",
                    "--hand", "AKo", "--position", "oop",
                    timeout=120)
        assert r["hero_bucket_idx"] >= 0
        assert len(r["action_labels"]) > 0
        assert len(r["hero_strategy"]) == 16  # K buckets
        # hero AKo should bet often (strong hand)
        bucket_strat = r["hero_strategy"][r["hero_bucket_idx"]]
        bet_freq = sum(p for lbl, p in zip(r["action_labels"], bucket_strat)
                       if not lbl.startswith("check") and lbl != "fold")
        assert bet_freq > 0.5, f"AKo should bet frequently, got {bet_freq:.2f}"

    def test_ip_river_with_responses(self):
        """IP hero — output should include ip_responses against each OOP action."""
        r = run_cli("solve_river.py", "3bet_called", "500", "AK7r",
                    "check/check/check/check",
                    "--turn", "Ts", "--river", "2h",
                    "--hand", "KQo", "--position", "ip",
                    timeout=120)
        assert r["hero_bucket_idx"] >= 0
        assert isinstance(r["ip_responses"], dict)
        # Should have a response entry for OOP's check at minimum
        assert any(path in r["ip_responses"] for path in ["check", "bet_11.88", "bet_18.00"])

    def test_ip_out_of_range_approximated(self):
        """KQo as IP in sr_called_ip_caller should trigger river approximation (flag is in table/context)."""
        r = run_cli("solve_river.py", "sr_called_ip_caller", "200", "K62r",
                    "check/check/check/check",
                    "--turn", "Ts", "--river", "2h",
                    "--hand", "KQo", "--position", "ip",
                    timeout=120)
        assert r["hero_bucket_idx"] >= 0
        assert "approximated" in r["context"]

    def test_vs_action_pin(self):
        r = run_cli("solve_river.py", "3bet_called", "500", "AK7r",
                    "check/check/check/check",
                    "--turn", "Ts", "--river", "2h",
                    "--hand", "KQo", "--position", "ip",
                    "--vs-action", "check",
                    timeout=120)
        assert r["vs_action"] == "check"

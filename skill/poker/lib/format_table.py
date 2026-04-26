"""Format solver output into the user's preferred {action, 频率, EV, 理由} table.

Caller is expected to:
1. Run solver via solve.solve(spec) — get full output dict
2. Identify the user's specific decision node (root, or deeper via action path)
3. Identify the user's specific hero combo (e.g. "9h9d")
4. Pass node + combo into format_decision_table to get a table-ready list of dicts
5. Add 理由 column (LLM prose, anchored on the freq/EV numbers)

The 理由 column is intentionally NOT auto-generated here — that's the LLM's job in the
skill flow. This module only handles the deterministic part.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def format_decision_table(
    solver_output: Dict[str, Any],
    hero_combo: str,
    decision_node_path: str = "",
    drop_below_pct: float = 1.0,
) -> List[Dict[str, Any]]:
    """Build the per-action rows for the decision table.

    Returns list of dicts: {"action": str, "freq_pct": float, "ev_chips": float | None}.
    EV is None unless the solver was instructed to compute per-action EVs (TODO — solver
    currently emits only frequencies and overall hero_value at root).

    `decision_node_path`: empty for root, e.g. "bet_101.50" for hero responding to villain bet.
    Requires solver_output["all_strategies"] to be populated.
    """
    if decision_node_path == "":
        # Root node — use solver_output["strategy"] (combo → action_label → prob)
        if hero_combo not in solver_output["strategy"]:
            raise ValueError(
                f"hero combo {hero_combo!r} not in solver root strategy. "
                f"Available combos: {list(solver_output['strategy'].keys())[:10]}..."
            )
        action_probs = solver_output["strategy"][hero_combo]  # dict
    else:
        # Walk into all_strategies for the right node
        all_strats = solver_output.get("all_strategies")
        if all_strats is None:
            raise ValueError(
                "decision_node_path requested but solver was not run with "
                "compute_all_strategies=true (or 'all_strategies' field missing)"
            )
        if decision_node_path not in all_strats:
            available = list(all_strats.keys())[:20]
            raise ValueError(
                f"decision_node_path {decision_node_path!r} not in all_strategies. "
                f"Sample paths: {available}"
            )
        node = all_strats[decision_node_path]
        # node["probabilities"] is [bucket_idx][action_idx]; bucket_idx aligns with hero_combos list
        hero_combos = solver_output.get("hero_combos") or solver_output.get("villain_combos")
        if hero_combos is None:
            raise ValueError("solver output missing 'hero_combos' / 'villain_combos'")
        # Determine which side this node belongs to (player), pick correct combo list
        player = node["player"]
        side_combos = solver_output["hero_combos"] if player == 0 else solver_output["villain_combos"]
        if hero_combo not in side_combos:
            raise ValueError(
                f"hero combo {hero_combo!r} not in this node's player combos "
                f"(player={player}). Available: {side_combos[:10]}..."
            )
        bucket_idx = side_combos.index(hero_combo)
        probs_list = node["probabilities"][bucket_idx]
        action_labels = node["actions"]
        action_probs = dict(zip(action_labels, probs_list))

    # Build rows, sorted by frequency desc
    rows: List[Dict[str, Any]] = []
    for action, p in action_probs.items():
        if p * 100.0 < drop_below_pct:
            continue
        rows.append({
            "action": action,
            "freq_pct": round(p * 100.0, 1),
            "ev_chips": None,  # solver doesn't expose per-action EV yet — TODO
        })
    rows.sort(key=lambda r: -r["freq_pct"])
    return rows


def render_markdown_table(
    rows: List[Dict[str, Any]],
    reasoning_per_action: Optional[Dict[str, str]] = None,
) -> str:
    """Render rows as a Markdown table with 理由 column."""
    reasoning_per_action = reasoning_per_action or {}
    lines = ["| action | 频率 | EV | 理由 |", "|---|---|---|---|"]
    for r in rows:
        ev = "—" if r["ev_chips"] is None else f"{r['ev_chips']:+.1f}"
        reason = reasoning_per_action.get(r["action"], "")
        lines.append(f"| {r['action']} | {r['freq_pct']}% | {ev} | {reason} |")
    return "\n".join(lines)


def find_subnode_path(
    solver_output: Dict[str, Any],
    villain_action_label: str,
) -> Optional[str]:
    """Helper: given the root villain action (e.g. 'bet_101.50'), return the action
    path string to use for hero's response node. Returns None if not found."""
    all_strats = solver_output.get("all_strategies", {})
    # Try exact match first
    if villain_action_label in all_strats:
        return villain_action_label
    # Try fuzzy match (closest by amount if it's a sized action)
    candidates = [k for k in all_strats if villain_action_label.split("_")[0] in k and " > " not in k]
    if not candidates:
        return None
    # Closest by parsed amount
    if "_" in villain_action_label:
        try:
            target = float(villain_action_label.split("_", 1)[1])
            candidates.sort(key=lambda k: abs(float(k.split("_", 1)[1]) - target) if "_" in k else 1e9)
            return candidates[0]
        except ValueError:
            pass
    return candidates[0]


if __name__ == "__main__":
    # Smoke test using a saved JSON
    import json, sys
    if len(sys.argv) < 3:
        print("Usage: format_table.py <solver_output.json> <hero_combo> [decision_path]")
        sys.exit(1)
    output = json.loads(open(sys.argv[1]).read())
    combo = sys.argv[2]
    path = sys.argv[3] if len(sys.argv) > 3 else ""
    rows = format_decision_table(output, combo, path)
    print(render_markdown_table(rows))

"""River on-demand solver: filter ranges along action path, then run bucketed CFR+.

Public API:
    solve_river(action_line, stack_bb, flop, action_path, river_card,
                hero_hand=None, position=None, iterations=500, k_buckets=16)
    → dict with "table", "context", "hero_strategy", "action_labels", ...
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .query_precompute import (
    find_flop_file, _load_json, _hand_type,
    ACTION_LINE_POTS, find_bucket_for_hand,
)

NUM_COMBOS = 1326

SOLVER_BINARY_NAME = "solve_river_subgame"


def _find_solver_binary() -> Path:
    root = Path(__file__).resolve().parents[3]
    for cand in [
        root / "target" / "release" / SOLVER_BINARY_NAME,
        root / "target" / "debug" / SOLVER_BINARY_NAME,
    ]:
        if cand.exists():
            import os
            if os.access(cand, os.X_OK):
                return cand
    raise FileNotFoundError(
        f"{SOLVER_BINARY_NAME} binary not found. Build with: "
        f"cd {root} && cargo build --release --bin {SOLVER_BINARY_NAME}"
    )


def _card_to_idx(card: str) -> int:
    """Convert card string like 'Ts', '2h', 'Kd' to 0-51 index."""
    RANK_ORDER = "23456789TJQKA"
    SUIT_ORDER = "shdc"
    r = RANK_ORDER.index(card[0].upper())
    s = SUIT_ORDER.index(card[1].lower())
    return r * 4 + s


def _initial_weights(bucket_of_combo: List[int]) -> List[float]:
    """Initial weights: 1.0 if combo is in range (bucket >= 0), else 0.0."""
    return [1.0 if b >= 0 else 0.0 for b in bucket_of_combo]


_COMBO_INDEX: Dict[Tuple[int, int], int] = {}


def _build_combo_index() -> None:
    """Build (card_a, card_b) → global combo index map matching Rust's
    combo_cards. Combos are enumerated as: for lo in 0..51, for hi in lo+1..52.
    """
    idx = 0
    for lo in range(51):
        for hi in range(lo + 1, 52):
            _COMBO_INDEX[(lo, hi)] = idx
            idx += 1


_build_combo_index()


def _combos_for_hand_type(ht: str) -> List[int]:
    """Return all global combo indices matching the given hand type (e.g. 'KQo')."""
    from .query_precompute import RANK_ORDER
    ht = ht.strip()
    if len(ht) == 2:   # pair, e.g. "JJ"
        r = RANK_ORDER.index(ht[0])
        out = []
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                a, b = r * 4 + s1, r * 4 + s2
                out.append(_COMBO_INDEX[(a, b)])
        return out
    r1 = RANK_ORDER.index(ht[0])
    r2 = RANK_ORDER.index(ht[1])
    suited = (len(ht) == 3 and ht[2].lower() == "s")
    out = []
    for s1 in range(4):
        for s2 in range(4):
            if r1 == r2:
                continue
            if suited and s1 != s2:
                continue
            if not suited and len(ht) == 3 and ht[2].lower() == "o" and s1 == s2:
                continue
            c1, c2 = r1 * 4 + s1, r2 * 4 + s2
            lo, hi = (c1, c2) if c1 < c2 else (c2, c1)
            out.append(_COMBO_INDEX[(lo, hi)])
    return sorted(set(out))


def _filter_weights_along_path(
    nodes: List[Dict],
    oop_weights: List[float],
    ip_weights: List[float],
    oop_bucket_of_combo: List[int],
    ip_bucket_of_combo: List[int],
    action_path: str,
    turn_card: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """Multiply combo weights by strategy probability at each decision node.

    For each action in the path:
      1. Find the node where that player is acting (player, current_path_prefix).
      2. Look up the action probability: strategy[bucket][action_idx].
      3. Multiply that player's weights by the probability.

    Returns (filtered_oop_weights, filtered_ip_weights).
    """
    if not action_path:
        return list(oop_weights), list(ip_weights)

    oop_w = list(oop_weights)
    ip_w  = list(ip_weights)

    # Build node lookup: (player, path, street, turn_card) → node
    node_map: Dict[Tuple, Dict] = {}
    for n in nodes:
        tc = n.get("turn_card", "").upper() if n.get("turn_card") else ""
        key = (n["player"], n["path"], n["street"], tc)
        node_map[key] = n

    actions = [a for a in action_path.split("/") if a]
    # Reconstruct the path prefix and which player is acting at each step
    # OOP (player 0) acts first each street; IP (player 1) responds.
    # Track player_to_act and path_so_far as we walk the actions.
    player = 0  # OOP acts first on flop
    path_so_far = ""
    street = "flop"
    # Once we hit the turn phase (after a check/check or call-check on flop),
    # street switches. We detect this by when turn_card is relevant.
    # For simplicity, check the node_map to find the right node.

    for action in actions:
        # Find the node where this player is acting at the current path
        tc_key = turn_card.upper() if (turn_card and street == "turn") else ""
        node = node_map.get((player, path_so_far, street, tc_key))

        if node is None:
            # Try other street (might be transitioning)
            for s in ("flop", "turn"):
                tc_k = turn_card.upper() if (turn_card and s == "turn") else ""
                node = node_map.get((player, path_so_far, s, tc_k))
                if node is not None:
                    street = s
                    break

        if node is not None:
            action_labels = node["action_labels"]
            strategy = node["strategy"]  # [bucket][action_idx]

            # Find action index
            action_idx = None
            for i, lbl in enumerate(action_labels):
                if lbl == action:
                    action_idx = i
                    break

            if action_idx is not None:
                # Multiply acting player's weights by P(action | bucket)
                if player == 0:
                    boc = oop_bucket_of_combo
                    w = oop_w
                else:
                    boc = ip_bucket_of_combo
                    w = ip_w

                for combo in range(NUM_COMBOS):
                    b = boc[combo]
                    if b < 0:
                        continue
                    prob = strategy[b][action_idx]
                    w[combo] *= prob

        # Advance state
        if path_so_far:
            path_so_far = path_so_far + "/" + action
        else:
            path_so_far = action

        # Determine next player based on action type
        if action in ("check", "call", "fold"):
            player = 1 - player
        elif action.startswith(("bet_", "raise_", "allin_")):
            player = 1 - player  # opponent now faces the bet

        # Detect street transition: after a check-check or call-ends-round,
        # the next street starts with OOP (player 0) again.
        # We reset player to 0 when the next street begins.
        # Heuristic: if player goes back to 0 after the action, we're on the
        # same logic (alternating). But after "call" or second "check",
        # the round ends and OOP leads the next street.
        # The simplest approach: look up the NEXT node in node_map.
        # If we can't find (player, path_so_far) but find (0, path_so_far), reset.
        if action in ("call",) or (action == "check" and player == 0):
            # End of a betting round — next street starts
            # Check if we're now on the turn
            if street == "flop" and turn_card is not None:
                street = "turn"
                player = 0  # OOP leads turn

    return oop_w, ip_w


def _pot_and_stacks_after_path(base_pot: float, stack_bb: float, action_path: str) -> Tuple[float, float, float]:
    """Compute (pot, oop_stack, ip_stack) after tracing the action path.

    Uses the tree builder convention:
      after_bet(extra): stack[me] -= extra, pot += extra, opp_to_call = extra - old_to_call
      after_call(to_call): stack[caller] -= to_call, pot += to_call
    """
    pot = base_pot
    oop_stack = float(stack_bb)
    ip_stack  = float(stack_bb)

    if not action_path:
        return pot, oop_stack, ip_stack

    actions = [a for a in action_path.split("/") if a]
    player = 0   # OOP acts first
    to_call = 0.0  # amount the current player faces

    for action in actions:
        if action == "check" or action == "fold":
            to_call = 0.0
            player = 1 - player
        elif action == "call":
            call_amt = min(to_call, oop_stack if player == 0 else ip_stack)
            if player == 0:
                oop_stack -= call_amt
            else:
                ip_stack -= call_amt
            pot += call_amt
            to_call = 0.0
            player = 1 - player
        elif action.startswith(("bet_", "raise_", "allin_")):
            extra = float(action.split("_", 1)[1])
            if player == 0:
                oop_stack -= extra
            else:
                ip_stack -= extra
            pot += extra
            # Opponent's new to_call = extra - what the current player was already facing
            to_call = max(0.0, extra - to_call)
            player = 1 - player

    return pot, oop_stack, ip_stack


def _format_river_table(
    action_labels: List[str],
    hero_strategy: List[List[float]],
    hero_bucket_idx: int,
    bucket_hands: Any,
    pot: float,
    hero_ehs_range: Optional[List] = None,
) -> str:
    """Render markdown strategy table for the river subgame."""
    if hero_bucket_idx < 0 or hero_bucket_idx >= len(hero_strategy):
        return f"(bucket {hero_bucket_idx} out of range)"

    probs = hero_strategy[hero_bucket_idx]
    hands_in_bucket = bucket_hands[hero_bucket_idx] if isinstance(bucket_hands, list) and hero_bucket_idx < len(bucket_hands) else []

    rows = []
    for label, p in zip(action_labels, probs):
        if p < 0.005:
            continue
        # Humanize label
        if label in ("check", "fold", "call"):
            human = label
        elif "_" in label:
            kind, amt_str = label.split("_", 1)
            try:
                amt = float(amt_str)
                if kind == "allin":
                    human = f"all-in ({amt:.0f} BB)"
                else:
                    pct = int(round(amt / pot * 100)) if pot > 0 else 0
                    human = f"{kind} {pct}% pot ({amt:.1f} BB)"
            except ValueError:
                human = label
        else:
            human = label
        rows.append((human, p))

    rows.sort(key=lambda r: -r[1])

    hands_str = ", ".join(hands_in_bucket[:6]) if hands_in_bucket else f"bucket {hero_bucket_idx}"
    if len(hands_in_bucket) > 6:
        hands_str += ", ..."

    ehs_str = ""
    if hero_ehs_range and hero_bucket_idx < len(hero_ehs_range):
        lo, hi = hero_ehs_range[hero_bucket_idx]
        ehs_str = f"  |  EHS {lo:.3f}–{hi:.3f}"

    lines = [
        f"**Pot:** {pot:.1f} BB  |  **Bucket {hero_bucket_idx}** ({hands_str}){ehs_str}",
        "",
        "| 操作 | 频率 | 理由 |",
        "|------|------|------|",
    ]
    for human, p in rows:
        lines.append(f"| {human} | {p*100:.0f}% | |")

    return "\n".join(lines)


def solve_river(
    action_line: str,
    stack_bb: int,
    flop: str,
    action_path: str,
    river_card: str,
    hero_hand: Optional[str] = None,
    position: Optional[str] = None,
    turn_card: Optional[str] = None,
    vs_action: Optional[str] = None,
    iterations: int = 500,
    k_buckets: int = 16,
) -> Dict[str, Any]:
    """Solve the river subgame from a given action path.

    Steps:
      1. Load precomputed flop file.
      2. Filter OOP and IP ranges along action_path using precomputed strategies.
      3. Build 5-card board (flop + turn_card + river_card).
      4. Call Rust solve_river_subgame binary.
      5. Return formatted strategy table + raw result.

    Args:
        action_line: "limped" | "sr_called" | "3bet_called" | "4bet_called"
        stack_bb: 200 or 500
        flop: e.g. "AK7r", "AhKd7c"
        action_path: e.g. "check/check" (flop), "check/check/check/check" (flop+turn cc)
        river_card: e.g. "2h", "Ts"
        hero_hand: optional, e.g. "AKo" — for showing hero's specific bucket strategy
        position: "oop" or "ip" (only needed if hero_hand given)
        turn_card: e.g. "Ts" — required if action_path includes turn actions
        iterations: CFR+ iterations (default 500)
        k_buckets: number of range buckets (default 16)
    """
    # Step 1: load precomputed file
    f = find_flop_file(action_line, stack_bb, flop)
    if f is None:
        raise FileNotFoundError(
            f"No precomputed file for {action_line}/{stack_bb}bb/{flop}."
        )
    data = _load_json(f)

    flop_label = data["flop_label"]
    oop_boc = data.get("oop_bucket_of_combo")
    ip_boc  = data.get("ip_bucket_of_combo")
    if oop_boc is None or ip_boc is None:
        raise ValueError(
            "oop_bucket_of_combo / ip_bucket_of_combo missing. "
            "Run: cargo run --release --bin patch_bucket_arrays"
        )

    # Step 2: filter ranges
    oop_initial = _initial_weights(oop_boc)
    ip_initial  = _initial_weights(ip_boc)

    # If hero's hand isn't in the precomputed range, inject its combos
    # at weight 1.0 so the Rust river solver re-buckets it on the actual
    # 5-card board (approximation — strategies at flop/turn nodes aren't
    # adjusted for this added range, so the filtering step leaves these
    # combos at weight 1.0 throughout).
    hero_hand_approximated = False
    if hero_hand and position:
        is_ip_hero = position.lower() in ("ip", "sb", "1")
        hero_boc = ip_boc if is_ip_hero else oop_boc
        target_ht = _hand_type(hero_hand)
        hero_combos = _combos_for_hand_type(target_ht)
        if all(hero_boc[c] < 0 for c in hero_combos):
            hero_hand_approximated = True
            hero_weights = ip_initial if is_ip_hero else oop_initial
            for c in hero_combos:
                hero_weights[c] = 1.0

    oop_w, ip_w = _filter_weights_along_path(
        nodes=data["nodes"],
        oop_weights=oop_initial,
        ip_weights=ip_initial,
        oop_bucket_of_combo=oop_boc,
        ip_bucket_of_combo=ip_boc,
        action_path=action_path,
        turn_card=turn_card,
    )

    # Step 3: build 5-card board
    flop_cards = [flop_label[i:i+2] for i in range(0, 6, 2)]
    if turn_card is None:
        raise ValueError("turn_card is required for river solve (need full 5-card board)")

    board_cards = flop_cards + [turn_card, river_card]
    try:
        board_indices = [_card_to_idx(c) for c in board_cards]
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid card in board {board_cards}: {e}")

    # Step 4: compute pot + stacks
    base_pot = ACTION_LINE_POTS.get(action_line, 12.0)
    pot, oop_stack, ip_stack = _pot_and_stacks_after_path(base_pot, stack_bb, action_path)

    # Zero out weights for combos that conflict with the board
    board_mask = 0
    for idx in board_indices:
        board_mask |= (1 << idx)

    from .query_precompute import RANK_ORDER as _RO
    # combo_cards lookup
    def combo_cards(ci):
        # global combo index to (card_a, card_b) - same logic as Rust
        # combos are indexed as: for a in 0..52 for b in 0..a → combo
        a = 0
        for aa in range(1, 52):
            for bb in range(aa):
                if a == ci:
                    return aa, bb
                a += 1
        return 0, 0

    # Build combo_cards lookup table once
    _combo_table = []
    ci = 0
    for aa in range(1, 52):
        for bb in range(aa):
            _combo_table.append((aa, bb))
            ci += 1

    for combo_i, (ca, cb) in enumerate(_combo_table):
        if board_mask & ((1 << ca) | (1 << cb)):
            oop_w[combo_i] = 0.0
            ip_w[combo_i]  = 0.0

    # Step 5: call Rust binary
    solver_input = {
        "board": board_indices,
        "oop_weights": oop_w,
        "ip_weights":  ip_w,
        "pot":        pot,
        "oop_stack":  max(0.0, oop_stack),
        "ip_stack":   max(0.0, ip_stack),
        "first_to_act": 0,  # OOP always leads river
        "iterations": iterations,
        "k_buckets":  k_buckets,
    }

    binary = _find_solver_binary()
    proc = subprocess.run(
        [str(binary)],
        input=json.dumps(solver_input),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"River solver failed:\n{proc.stderr.strip()}"
        )
    result = json.loads(proc.stdout)

    # Step 6: pick the right decision node for hero
    # - OOP (player=0) acts at the river root → path=""
    # - IP (player=1) responds to OOP action → path="check" or "bet_X" etc.
    is_ip = position and position.lower() in ("ip", "sb", "1")
    nodes = result.get("nodes", [])
    hero_player = 1 if is_ip else 0

    # Collect matching nodes for hero's side
    hero_nodes = [n for n in nodes if n["player"] == hero_player]

    # Pick the primary node to display
    primary_node = None
    if not is_ip:
        # OOP at root (path="")
        for n in hero_nodes:
            if n["path"] == "":
                primary_node = n
                break
    else:
        # IP: if vs_action provided, find that node. Otherwise pick the most
        # likely OOP action (by root-strategy frequency summed across buckets).
        if vs_action:
            for n in hero_nodes:
                if n["path"] == vs_action:
                    primary_node = n
                    break
        else:
            # Pick IP's response to OOP's most-frequent action
            oop_root = next((n for n in nodes if n["player"] == 0 and n["path"] == ""), None)
            if oop_root:
                # Sum probability across buckets for each action
                action_totals = [0.0] * len(oop_root["action_labels"])
                for bucket_probs in oop_root["strategy"]:
                    for i, p in enumerate(bucket_probs):
                        action_totals[i] += p
                # Pick the highest-frequency action that has a matching IP response
                order = sorted(range(len(action_totals)), key=lambda i: -action_totals[i])
                for idx in order:
                    candidate_path = oop_root["action_labels"][idx]
                    for n in hero_nodes:
                        if n["path"] == candidate_path:
                            primary_node = n
                            vs_action = candidate_path
                            break
                    if primary_node:
                        break

    # Find hero bucket
    hero_bucket_idx = -1
    hero_bucket_hands: List[str] = []
    if hero_hand and position:
        ht = _hand_type(hero_hand)
        bucket_hands_list = result.get(
            "villain_bucket_hands" if is_ip else "bucket_hands", []
        )
        for bi, hands in enumerate(bucket_hands_list):
            if isinstance(hands, list) and ht in hands:
                hero_bucket_idx = bi
                hero_bucket_hands = hands
                break

    # Format primary table
    if primary_node is not None:
        display_bucket = hero_bucket_idx if hero_bucket_idx >= 0 else 0
        ehs_range = result.get("villain_ehs_range" if is_ip else "hero_ehs_range", [])
        table_md = _format_river_table(
            action_labels=primary_node["action_labels"],
            hero_strategy=primary_node["strategy"],
            hero_bucket_idx=display_bucket,
            bucket_hands=result.get("villain_bucket_hands" if is_ip else "bucket_hands", []),
            pot=pot,
            hero_ehs_range=ehs_range,
        )
    else:
        table_md = "(no matching decision node found)"

    # Build IP response summary table (all OOP actions → IP's response)
    ip_responses: Dict[str, Dict[str, Any]] = {}
    if is_ip and hero_bucket_idx >= 0:
        for n in hero_nodes:
            if "/" in n["path"]:
                continue  # only first-level IP responses
            ip_responses[n["path"]] = {
                "action_labels": n["action_labels"],
                "strategy": n["strategy"][hero_bucket_idx] if hero_bucket_idx < len(n["strategy"]) else [],
            }

    pos_label = ""
    if hero_hand and position:
        pos_name = "IP/SB" if is_ip else "OOP/BB"
        pos_label = f"  |  **Hero:** {_hand_type(hero_hand)} ({pos_name})"
        if hero_hand_approximated:
            pos_label += " *(approximated — not in precomputed range)*"
        elif hero_bucket_idx < 0:
            pos_label += " *(hand not in filtered range)*"
        if is_ip and vs_action:
            pos_label += f"  |  **Facing:** OOP {vs_action}"

    context = (
        f"**Board:** {flop_label} {turn_card} {river_card}  |  "
        f"**Line:** {action_line}/{stack_bb}bb  |  "
        f"**Pot:** {pot:.1f} BB  |  "
        f"**Effective stack:** {min(oop_stack, ip_stack):.1f} BB"
        f"{pos_label}"
    )

    return {
        "table": table_md,
        "context": context,
        "flop_label": flop_label,
        "board": board_cards,
        "pot": pot,
        "hero_strategy": (primary_node["strategy"] if primary_node else result["hero_strategy"]),
        "action_labels": (primary_node["action_labels"] if primary_node else result["action_labels"]),
        "bucket_hands": result.get("bucket_hands", []),
        "villain_bucket_hands": result.get("villain_bucket_hands", []),
        "hero_ehs_range": result.get("hero_ehs_range", []),
        "villain_ehs_range": result.get("villain_ehs_range", []),
        "hero_value": result["hero_value"],
        "iterations": result["iterations"],
        "hero_bucket_idx": hero_bucket_idx,
        "vs_action": vs_action if is_ip else None,
        "ip_responses": ip_responses,
        "nodes": nodes,
    }

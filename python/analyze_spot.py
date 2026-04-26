from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .analyze_kuhn import analyze_kuhn
from .analyze_leduc import analyze_leduc
from .nlhe.analyze import analyze_river_spot
from .nlhe.equity import river_equity_vs_range


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic spot analyzer entrypoint.")
    parser.add_argument("--input-file", required=True, help="Path to a JSON input file")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    return parser


def load_payload(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    game = payload.get("game")
    if game == "kuhn":
        return analyze_kuhn(
            hero_card=payload["hero_card"],
            history=payload.get("history", ""),
            iterations=int(payload.get("iterations", 20000)),
        )
    if game == "leduc":
        round_histories = payload.get("round_histories", ["", ""])
        return analyze_leduc(
            hero_card=payload["hero_card"],
            public_card=payload.get("public_card"),
            round_histories=(round_histories[0], round_histories[1]),
            iterations=int(payload.get("iterations", 50)),
        )
    if game == "nlhe_river":
        return river_equity_vs_range(
            board=payload["board"],
            hero_hand=payload["hero_hand"],
            pot=float(payload["pot"]),
            to_call=float(payload["to_call"]),
            villain_range=payload["villain_range"],
        )
    if game == "nlhe_flop_solve":
        from .nlhe.tree import build_flop_tree
        from .nlhe.flop_best_response import best_response_value_flop
        from .nlhe.flop_cfr import build_and_train_flop
        stacks = payload["stacks"]
        board_3 = payload["board"]
        if len(board_3) != 3:
            raise ValueError("nlhe_flop_solve requires exactly 3 board cards")
        hero_range = _parse_range_str(payload["hero_range"])
        villain_range = _parse_range_str(payload["villain_range"])
        iterations = int(payload.get("iterations", 30))
        max_raises = int(payload.get("max_raises", 1))
        turn_max_raises = int(payload.get("turn_max_raises", 1))
        river_max_raises = int(payload.get("river_max_raises", 1))
        first_to_act = int(payload.get("first_to_act", 0))
        tree = build_flop_tree(
            board_3=board_3,
            pot=float(payload["pot"]),
            stacks=(float(stacks[0]), float(stacks[1])),
            first_to_act=first_to_act,
            max_raises=max_raises,
            turn_max_raises=turn_max_raises,
            river_max_raises=river_max_raises,
        )
        solver = build_and_train_flop(
            tree, hero_range, villain_range, board_3, iterations,
        )
        out = {
            "game": "nlhe_flop_solve",
            "board": list(board_3),
            "pot": float(payload["pot"]),
            "stacks": [float(stacks[0]), float(stacks[1])],
            "first_to_act": first_to_act,
            "acting_player": "hero" if first_to_act == 0 else "villain",
            "iterations": iterations,
            "max_raises": max_raises,
            "turn_max_raises": turn_max_raises,
            "river_max_raises": river_max_raises,
            "hero_ev": solver.last_root_value(),
            "strategy": {
                _combo_idx_to_label(c): dict(probs)
                for c, probs in solver.extract_root_strategy().items()
            },
        }
        if bool(payload.get("compute_exploitability", False)):
            br_h = best_response_value_flop(solver, 0)
            br_v = best_response_value_flop(solver, 1)
            out["exploitability"] = {
                "br_hero": br_h, "br_villain": br_v,
                "initial_pot": solver.root.pot,
                "exploitability": br_h + br_v - solver.root.pot,
            }
        return out

    if game == "nlhe_turn_solve":
        from .nlhe.tree import build_turn_tree
        from .nlhe.turn_best_response import best_response_value_turn
        from .nlhe.turn_cfr import build_and_train_turn
        stacks = payload["stacks"]
        board_4 = payload["board"]
        if len(board_4) != 4:
            raise ValueError("nlhe_turn_solve requires exactly 4 board cards")
        hero_range = _parse_range_str(payload["hero_range"])
        villain_range = _parse_range_str(payload["villain_range"])
        iterations = int(payload.get("iterations", 150))
        max_raises = int(payload.get("max_raises", 2))
        river_max_raises = int(payload.get("river_max_raises", 2))
        first_to_act = int(payload.get("first_to_act", 0))
        tree = build_turn_tree(
            board_4=board_4,
            pot=float(payload["pot"]),
            stacks=(float(stacks[0]), float(stacks[1])),
            first_to_act=first_to_act,
            max_raises=max_raises,
            river_max_raises=river_max_raises,
        )
        solver = build_and_train_turn(
            tree, hero_range, villain_range, board_4, iterations,
        )
        root_strategy = solver.extract_root_strategy()
        out = {
            "game": "nlhe_turn_solve",
            "board": list(board_4),
            "pot": float(payload["pot"]),
            "stacks": [float(stacks[0]), float(stacks[1])],
            "first_to_act": first_to_act,
            "acting_player": "hero" if first_to_act == 0 else "villain",
            "iterations": iterations,
            "max_raises": max_raises,
            "river_max_raises": river_max_raises,
            "hero_ev": solver.last_root_value(),
            "strategy": {
                _combo_idx_to_label(c): dict(probs) for c, probs in root_strategy.items()
            },
        }
        hero_hand_raw = payload.get("hero_hand")
        if isinstance(hero_hand_raw, str):
            hero_hand_raw = _split_hand_string(hero_hand_raw)
        if hero_hand_raw is not None:
            from .nlhe.cards import combo_index
            hi = combo_index(hero_hand_raw[0], hero_hand_raw[1])
            out["hero_hand"] = _combo_idx_to_label(hi)
            if first_to_act == 0 and hi in root_strategy:
                probs = root_strategy[hi]
                sorted_actions = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
                out["recommendation"] = {
                    "hand": _combo_idx_to_label(hi),
                    "top_action": sorted_actions[0][0],
                    "top_probability": sorted_actions[0][1],
                    "actions": [{"action": a, "probability": p} for a, p in sorted_actions],
                }
        if bool(payload.get("compute_exploitability", False)):
            br_h = best_response_value_turn(solver, 0)
            br_v = best_response_value_turn(solver, 1)
            out["exploitability"] = {
                "br_hero": br_h,
                "br_villain": br_v,
                "initial_pot": solver.root.pot,
                "exploitability": br_h + br_v - solver.root.pot,
            }
        return out

    if game == "nlhe_river_solve_rust":
        from .nlhe.rust_bridge import solve_via_rust
        return solve_via_rust({
            "game": "river",
            "board": payload["board"],
            "pot": float(payload["pot"]),
            "stacks": payload["stacks"],
            "first_to_act": int(payload.get("first_to_act", 0)),
            "hero_range": payload["hero_range"],
            "villain_range": payload["villain_range"],
            "iterations": int(payload.get("iterations", 500)),
            "max_raises": int(payload.get("max_raises", 2)),
            "compute_exploitability": bool(payload.get("compute_exploitability", False)),
        })
    if game == "nlhe_turn_solve_rust":
        from .nlhe.rust_bridge import solve_via_rust
        return solve_via_rust({
            "game": "turn",
            "board": payload["board"],
            "pot": float(payload["pot"]),
            "stacks": payload["stacks"],
            "first_to_act": int(payload.get("first_to_act", 0)),
            "hero_range": payload["hero_range"],
            "villain_range": payload["villain_range"],
            "iterations": int(payload.get("iterations", 200)),
            "max_raises": int(payload.get("max_raises", 1)),
            "river_max_raises": int(payload.get("river_max_raises", 1)),
            "compute_exploitability": bool(payload.get("compute_exploitability", False)),
        })
    if game == "nlhe_flop_solve_rust":
        from .nlhe.rust_bridge import solve_via_rust
        return solve_via_rust({
            "game": "flop",
            "board": payload["board"],
            "pot": float(payload["pot"]),
            "stacks": payload["stacks"],
            "first_to_act": int(payload.get("first_to_act", 0)),
            "hero_range": payload["hero_range"],
            "villain_range": payload["villain_range"],
            "iterations": int(payload.get("iterations", 50)),
            "max_raises": int(payload.get("max_raises", 1)),
            "turn_max_raises": int(payload.get("turn_max_raises", 1)),
            "river_max_raises": int(payload.get("river_max_raises", 1)),
            "compute_exploitability": bool(payload.get("compute_exploitability", False)),
        })
    if game == "nlhe_river_solve":
        stacks = payload["stacks"]
        hero_hand = payload.get("hero_hand")
        if isinstance(hero_hand, str):
            hero_hand = _split_hand_string(hero_hand)
        return analyze_river_spot(
            board=payload["board"],
            pot=float(payload["pot"]),
            stacks=(float(stacks[0]), float(stacks[1])),
            hero_range=payload["hero_range"],
            villain_range=payload["villain_range"],
            first_to_act=int(payload.get("first_to_act", 0)),
            hero_hand=hero_hand,
            iterations=int(payload.get("iterations", 500)),
            max_raises=int(payload.get("max_raises", 2)),
            n_buckets=payload.get("n_buckets"),
            compute_exploitability=bool(payload.get("compute_exploitability", False)),
        )
    raise SystemExit(f"unsupported game: {game}")


def _split_hand_string(hand: str) -> list:
    """Turn 'AsAc' into ['As', 'Ac']."""
    if len(hand) != 4:
        raise ValueError(f"hero_hand string must be 4 chars like 'AsAc', got {hand!r}")
    return [hand[0:2], hand[2:4]]


def _parse_range_str(r) -> list:
    """Accept either a range string or a pre-parsed 1326-dim list."""
    from .nlhe.range_parser import parse_range
    if isinstance(r, str):
        return parse_range(r)
    return list(r)


def _combo_idx_to_label(idx: int) -> str:
    from .nlhe.cards import INDEX_TO_CARD, INDEX_TO_COMBO
    a, b = INDEX_TO_COMBO[idx]
    return f"{INDEX_TO_CARD[a]}{INDEX_TO_CARD[b]}"


def print_text_result(result: Dict[str, Any]) -> None:
    if result.get("game") == "nlhe_river_solve":
        _print_river_solve(result)
        return
    print(f"game: {result['game']}")
    if result["game"] in {"kuhn", "leduc"}:
        print(f"hero_card: {result['hero_card']}")
    if result["game"] == "kuhn":
        print(f"history: {result['history'] or '<root>'}")
    elif result["game"] == "leduc":
        public_card = result["public_card"] or "-"
        print(f"public_card: {public_card}")
        print(f"round_histories: {result['round_histories']}")
    elif result["game"] == "nlhe_river":
        print(f"board: {result['board']}")
        print(f"hero_hand: {result['hero_hand']}")
        print(f"pot: {result['pot']:.2f}")
        print(f"to_call: {result['to_call']:.2f}")
        print(f"villain_combo_count: {result['villain_combo_count']}")
    print(f"recommended_action: {result['recommended_action']}")
    if "strategy" in result:
        for action, probability in result["strategy"].items():
            print(f"  {action}: {probability:.4f}")
    print(f"player_0_value: {result['player_0_value']:.6f}")
    if "equity" in result:
        print(f"equity: {result['equity']:.6f}")
    if "required_equity" in result:
        print(f"required_equity: {result['required_equity']:.6f}")
    if "ev_call" in result:
        print(f"ev_call: {result['ev_call']:.6f}")
    if "training_game_value" in result:
        print(f"training_game_value: {result['training_game_value']:.6f}")
    if "best_response_player_0" in result:
        print(f"best_response_player_0: {result['best_response_player_0']:.6f}")
    if "best_response_player_1" in result:
        print(f"best_response_player_1: {result['best_response_player_1']:.6f}")
    if "exploitability" in result:
        print(f"exploitability: {result['exploitability']:.6f}")
    if "infoset_count" in result:
        print(f"infoset_count: {result['infoset_count']}")


def _print_river_solve(result: Dict[str, Any]) -> None:
    print(f"game: {result['game']}")
    print(f"board: {' '.join(result['board'])}")
    print(f"pot: {result['pot']:.2f}  stacks: {result['stacks']}  first_to_act: {result['first_to_act']} ({result['acting_player']})")
    print(f"iterations: {result['iterations']}  max_raises: {result['max_raises']}")
    print(f"hero_range_combos: {result['hero_range_combos']}  villain_range_combos: {result['villain_range_combos']}")
    print(f"hero_ev: {result['hero_ev']:.4f}")
    print()
    print(f"strategy for {result['acting_player']} ({result['acting_range']}):")
    for hand, probs in result["strategy"].items():
        nonzero = [(a, p) for a, p in probs.items() if p >= 0.005]
        nonzero.sort(key=lambda kv: kv[1], reverse=True)
        parts = "  ".join(f"{a}={p:.2%}" for a, p in nonzero) or "<uniform>"
        print(f"  {hand}: {parts}")
    exp_info = result.get("exploitability")
    if exp_info is not None:
        print()
        print(f"exploitability: {exp_info['exploitability']:.3f} (BR_h={exp_info['br_hero']:.3f}, BR_v={exp_info['br_villain']:.3f}, pot={exp_info['initial_pot']:.3f})")
        print("  (0 = GTO; higher = further from Nash; always >= 0)")
    rec = result.get("recommendation")
    if rec is not None:
        print()
        if "note" in rec:
            print(f"note: {rec['note']}")
        else:
            print(f"recommendation for {rec['hand']}:")
            print(f"  top: {rec['top_action']} ({rec['top_probability']:.2%})")
            for a in rec["actions"]:
                print(f"    {a['action']}: {a['probability']:.2%}")


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_payload(load_payload(args.input_file))
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False))
        return

    print_text_result(result)


if __name__ == "__main__":
    main()

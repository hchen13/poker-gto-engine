from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .analyze_kuhn import analyze_kuhn
from .analyze_leduc import analyze_leduc
from .analyze_nlhe_river import analyze_nlhe_river


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
        return analyze_nlhe_river(
            board=payload["board"],
            hero_hand=payload["hero_hand"],
            pot=float(payload["pot"]),
            to_call=float(payload["to_call"]),
            villain_range=payload["villain_range"],
        )
    raise SystemExit(f"unsupported game: {game}")


def print_text_result(result: Dict[str, Any]) -> None:
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


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_payload(load_payload(args.input_file))
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False))
        return

    print_text_result(result)


if __name__ == "__main__":
    main()

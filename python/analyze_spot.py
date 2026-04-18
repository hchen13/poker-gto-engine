from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .analyze_kuhn import analyze_kuhn
from .analyze_leduc import analyze_leduc


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
    raise SystemExit(f"unsupported game: {game}")


def print_text_result(result: Dict[str, Any]) -> None:
    print(f"game: {result['game']}")
    print(f"hero_card: {result['hero_card']}")
    if result["game"] == "kuhn":
        print(f"history: {result['history'] or '<root>'}")
    elif result["game"] == "leduc":
        public_card = result["public_card"] or "-"
        print(f"public_card: {public_card}")
        print(f"round_histories: {result['round_histories']}")
    print(f"recommended_action: {result['recommended_action']}")
    for action, probability in result["strategy"].items():
        print(f"  {action}: {probability:.4f}")
    print(f"player_0_value: {result['player_0_value']:.6f}")
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

from __future__ import annotations

import argparse
import json
from typing import Dict

from .kuhn_cfr import train_kuhn_cfr

VALID_CARDS = {"J", "Q", "K"}
VALID_HISTORY_CHARS = {"p", "b"}
ACTION_NAMES = {"check": "check", "bet": "bet"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a Kuhn Poker information set using CFR.")
    parser.add_argument("--hero-card", required=True, help="Hero private card: J, Q, or K")
    parser.add_argument("--history", default="", help="Action history encoded with p(check/pass) and b(bet/call)")
    parser.add_argument("--iterations", type=int, default=20000, help="Training iterations")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    return parser


def validate_inputs(hero_card: str, history: str) -> None:
    if hero_card not in VALID_CARDS:
        raise SystemExit(f"invalid hero card: {hero_card}")
    invalid = sorted(set(history) - VALID_HISTORY_CHARS)
    if invalid:
        raise SystemExit(f"invalid history chars: {''.join(invalid)}")


def normalize_strategy(strategy: Dict[str, float]) -> Dict[str, float]:
    return {ACTION_NAMES[action]: probability for action, probability in strategy.items()}


def analyze_kuhn(hero_card: str, history: str, iterations: int) -> Dict[str, object]:
    validate_inputs(hero_card, history)
    summary = train_kuhn_cfr(iterations=iterations)
    infoset_key = hero_card + history
    if infoset_key not in summary["infoset_strategy"]:
        raise SystemExit(f"unknown information set: {infoset_key}")

    strategy = normalize_strategy(summary["infoset_strategy"][infoset_key])
    recommended_action = max(strategy, key=strategy.get)
    return {
        "game": "kuhn",
        "hero_card": hero_card,
        "history": history,
        "recommended_action": recommended_action,
        "strategy": strategy,
        "player_0_value": summary["player_0_value"],
    }


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_kuhn(args.hero_card, args.history, args.iterations)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False))
        return

    print(f"game: {result['game']}")
    print(f"hero_card: {result['hero_card']}")
    print(f"history: {result['history'] or '<root>'}")
    print(f"recommended_action: {result['recommended_action']}")
    for action, probability in result["strategy"].items():
        print(f"  {action}: {probability:.4f}")
    print(f"player_0_value: {result['player_0_value']:.6f}")


if __name__ == "__main__":
    main()

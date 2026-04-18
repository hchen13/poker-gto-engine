from __future__ import annotations

import argparse
import json
from typing import Dict, Tuple

from .leduc_cfr import train_leduc_cfr

VALID_RANKS = {"J", "Q", "K"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a Leduc Poker information set using CFR.")
    parser.add_argument("--hero-card", required=True, help="Hero private rank: J, Q, or K")
    parser.add_argument("--public-card", default="-", help="Public rank: J, Q, K, or - if unrevealed")
    parser.add_argument("--round-one-history", default="", help="Round one history encoded with x/b/c/r/f")
    parser.add_argument("--round-two-history", default="", help="Round two history encoded with x/b/c/r/f")
    parser.add_argument("--iterations", type=int, default=50, help="Training iterations")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    return parser


def validate_inputs(hero_card: str, public_card: str) -> None:
    if hero_card not in VALID_RANKS:
        raise SystemExit(f"invalid hero card: {hero_card}")
    if public_card != "-" and public_card not in VALID_RANKS:
        raise SystemExit(f"invalid public card: {public_card}")


def analyze_leduc(
    hero_card: str,
    public_card: str | None,
    round_histories: Tuple[str, str],
    iterations: int,
) -> Dict[str, object]:
    public_rank = None if public_card in (None, "-") else public_card
    validate_inputs(hero_card, public_rank or "-")
    summary = train_leduc_cfr(iterations=iterations)
    infoset_key = f"{hero_card}|{public_rank or '-'}|{round_histories[0]}|{round_histories[1]}"
    root_strategy = summary["root_strategy"] if public_rank is None and round_histories == ("", "") else None
    strategy = root_strategy[hero_card] if root_strategy is not None else None
    if strategy is None:
        raise SystemExit(f"unsupported leduc infoset for now: {infoset_key}")

    recommended_action = max(strategy, key=strategy.get)
    return {
        "game": "leduc",
        "hero_card": hero_card,
        "public_card": public_rank,
        "round_histories": list(round_histories),
        "recommended_action": recommended_action,
        "strategy": strategy,
        "player_0_value": summary["player_0_value"],
        "infoset_count": summary["infoset_count"],
    }


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_leduc(
        hero_card=args.hero_card,
        public_card=args.public_card,
        round_histories=(args.round_one_history, args.round_two_history),
        iterations=args.iterations,
    )
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False))
        return

    print(f"game: {result['game']}")
    print(f"hero_card: {result['hero_card']}")
    print(f"public_card: {result['public_card'] or '-'}")
    print(f"round_histories: {result['round_histories']}")
    print(f"recommended_action: {result['recommended_action']}")
    for action, probability in result["strategy"].items():
        print(f"  {action}: {probability:.4f}")
    print(f"player_0_value: {result['player_0_value']:.6f}")
    print(f"infoset_count: {result['infoset_count']}")


if __name__ == "__main__":
    main()

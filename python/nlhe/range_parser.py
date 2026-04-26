"""Parse poker range strings into combo-weight vectors.

Supported syntax (case-insensitive for ranks, suits lowercase):

    AA              pocket pair, all 6 combos
    AKs             suited hand class, 4 combos
    AKo             offsuit hand class, 12 combos
    AK              suited + offsuit combined, 16 combos
    TT+             pocket pair plus: TT, JJ, QQ, KK, AA
    88-TT           pocket pair range: 88, 99, TT (inclusive)
    A2s+            suited connector-style plus: A2s..AKs
    A5o+            offsuit plus: A5o..AKo
    T9s-76s         descending suited run: T9s, 98s, 87s, 76s
    AhKs            specific combo (explicit cards)
    AA:0.5          weight override (default weight is 1.0)

Multiple tokens are separated by commas or whitespace. Later tokens that mention
the same combo override earlier ones (so you can build a range additively and
carve out exceptions at the end).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .cards import (
    INDEX_TO_COMBO,
    NUM_COMBOS,
    RANK_TO_VALUE,
    SUITS,
    VALID_SUITS,
    combo_index,
)

WeightVector = List[float]


class RangeParseError(ValueError):
    pass


_TOKEN_SPLIT = re.compile(r"[,\s]+")


def parse_range(text: str) -> WeightVector:
    weights: Dict[int, float] = {}
    for raw in _TOKEN_SPLIT.split(text.strip()):
        if not raw:
            continue
        token, weight = _split_weight(raw)
        combos = _expand_token(token)
        if not combos:
            raise RangeParseError(f"token matched no combos: {raw!r}")
        for idx in combos:
            weights[idx] = weight

    vec = [0.0] * NUM_COMBOS
    for idx, w in weights.items():
        vec[idx] = w
    return vec


def range_to_combos(weights: WeightVector) -> List[Tuple[Tuple[str, str], float]]:
    from .cards import INDEX_TO_CARD

    out = []
    for idx, w in enumerate(weights):
        if w > 0:
            a, b = INDEX_TO_COMBO[idx]
            out.append(((INDEX_TO_CARD[a], INDEX_TO_CARD[b]), w))
    return out


def range_weight_total(weights: WeightVector) -> float:
    return sum(weights)


def _split_weight(raw: str) -> Tuple[str, float]:
    if ":" in raw:
        token, w = raw.rsplit(":", 1)
        try:
            weight = float(w)
        except ValueError as e:
            raise RangeParseError(f"invalid weight in {raw!r}") from e
        if not 0.0 <= weight <= 1.0:
            raise RangeParseError(f"weight must be in [0, 1], got {weight} in {raw!r}")
        return token.strip(), weight
    return raw.strip(), 1.0


def _expand_token(token: str) -> List[int]:
    if _looks_like_specific_combo(token):
        return [_parse_specific_combo(token)]

    if "-" in token:
        return _expand_dash_range(token)

    if token.endswith("+"):
        return _expand_plus(token[:-1])

    return _expand_class(token)


def _looks_like_specific_combo(token: str) -> bool:
    if len(token) != 4:
        return False
    return token[1] in VALID_SUITS and token[3] in VALID_SUITS


def _parse_specific_combo(token: str) -> int:
    a, b = token[:2], token[2:]
    r1, s1 = a[0].upper(), a[1].lower()
    r2, s2 = b[0].upper(), b[1].lower()
    if r1 not in RANK_TO_VALUE or r2 not in RANK_TO_VALUE:
        raise RangeParseError(f"invalid rank in specific combo {token!r}")
    if s1 not in VALID_SUITS or s2 not in VALID_SUITS:
        raise RangeParseError(f"invalid suit in specific combo {token!r}")
    return combo_index(f"{r1}{s1}", f"{r2}{s2}")


def _expand_class(token: str) -> List[int]:
    """Expand a hand-class token like 'AA', 'AKs', 'AKo', or 'AK'."""
    if len(token) not in (2, 3):
        raise RangeParseError(f"unrecognized token: {token!r}")

    r1 = token[0].upper()
    r2 = token[1].upper()
    if r1 not in RANK_TO_VALUE or r2 not in RANK_TO_VALUE:
        raise RangeParseError(f"invalid rank in token: {token!r}")

    suited = offsuit = pair = False
    if r1 == r2:
        pair = True
        if len(token) != 2:
            raise RangeParseError(f"pair token cannot carry suit marker: {token!r}")
    else:
        if len(token) == 3:
            m = token[2].lower()
            if m == "s":
                suited = True
            elif m == "o":
                offsuit = True
            else:
                raise RangeParseError(f"invalid suit marker: {token!r}")
        else:
            suited = True
            offsuit = True
        if RANK_TO_VALUE[r1] < RANK_TO_VALUE[r2]:
            r1, r2 = r2, r1

    indices: List[int] = []
    if pair:
        for i, s1 in enumerate(SUITS):
            for s2 in SUITS[i + 1 :]:
                indices.append(combo_index(f"{r1}{s1}", f"{r1}{s2}"))
    else:
        for s1 in SUITS:
            for s2 in SUITS:
                if suited and s1 == s2:
                    indices.append(combo_index(f"{r1}{s1}", f"{r2}{s2}"))
                if offsuit and s1 != s2:
                    indices.append(combo_index(f"{r1}{s1}", f"{r2}{s2}"))
    return indices


def _expand_plus(base: str) -> List[int]:
    """Expand 'TT+', 'A2s+', etc. — all hands 'as strong or stronger' in the usual sense."""
    if len(base) == 2 and base[0].upper() == base[1].upper():
        low = RANK_TO_VALUE[base[0].upper()]
        out: List[int] = []
        for v in range(low, 15):
            out.extend(_expand_class(f"{_v2r(v)}{_v2r(v)}"))
        return out

    if len(base) != 3:
        raise RangeParseError(f"invalid '+' token: {base}+")
    high, low, marker = base[0].upper(), base[1].upper(), base[2].lower()
    if high == low:
        raise RangeParseError(f"invalid '+' token (pair cannot have suit marker): {base}+")
    if marker not in ("s", "o"):
        raise RangeParseError(f"invalid suit marker in '+' token: {base}+")
    high_v = RANK_TO_VALUE[high]
    low_v = RANK_TO_VALUE[low]
    if low_v >= high_v:
        raise RangeParseError(f"invalid '+' token (low must be < high): {base}+")

    out = []
    for v in range(low_v, high_v):
        out.extend(_expand_class(f"{high}{_v2r(v)}{marker}"))
    return out


def _expand_dash_range(token: str) -> List[int]:
    left, right = [s.strip() for s in token.split("-", 1)]

    if (
        len(left) == 2
        and len(right) == 2
        and left[0] == left[1]
        and right[0] == right[1]
    ):
        a = RANK_TO_VALUE[left[0].upper()]
        b = RANK_TO_VALUE[right[0].upper()]
        lo, hi = (a, b) if a <= b else (b, a)
        out: List[int] = []
        for v in range(lo, hi + 1):
            out.extend(_expand_class(f"{_v2r(v)}{_v2r(v)}"))
        return out

    if len(left) == 3 and len(right) == 3 and left[2] == right[2] and left[2] in ("s", "o"):
        marker = left[2]
        l_high, l_low = left[0].upper(), left[1].upper()
        r_high, r_low = right[0].upper(), right[1].upper()
        l_gap = RANK_TO_VALUE[l_high] - RANK_TO_VALUE[l_low]
        r_gap = RANK_TO_VALUE[r_high] - RANK_TO_VALUE[r_low]
        if l_gap != r_gap:
            raise RangeParseError(
                f"dash range must preserve gap between ranks: {token!r}"
            )
        high_v_l = RANK_TO_VALUE[l_high]
        high_v_r = RANK_TO_VALUE[r_high]
        lo_high, hi_high = (high_v_r, high_v_l) if high_v_r <= high_v_l else (high_v_l, high_v_r)
        out = []
        for hv in range(lo_high, hi_high + 1):
            lv = hv - l_gap
            out.extend(_expand_class(f"{_v2r(hv)}{_v2r(lv)}{marker}"))
        return out

    raise RangeParseError(f"unrecognized dash range: {token!r}")


def _v2r(v: int) -> str:
    for rank, val in RANK_TO_VALUE.items():
        if val == v:
            return rank
    raise ValueError(f"invalid rank value: {v}")

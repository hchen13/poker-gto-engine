"""River HU game tree builder.

Scope: single street (river), heads-up. No chance nodes — all cards are already
known by the time we solve. The tree only branches on player actions.

Terminology:
- ``pot``: chips already in the middle at the start of the current decision
  (everything both players have committed on every prior street plus whatever
  has gone in on this street up to the last completed action).
- ``to_call``: how many chips the player to act owes to match the outstanding
  bet (0 if no one has bet on this street yet after the check/open-action).
- ``stacks``: remaining chips behind for each player. Index matches player id.
- ``player_to_act``: 0 or 1. Whoever is first to act on the river (by position)
  is player 0 here by convention — actual seat/position labels are attached
  outside the tree.

Terminal payoff is reported as (pot_total, winner_indicator):
- ``winner`` = 0 or 1 for a fold (that player wins the whole pot).
- ``winner`` = None for a showdown: solver must compute equity externally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BET_FRACTIONS: Tuple[float, ...] = (0.33, 0.50, 0.67, 1.00, 1.50)
RAISE_POT_FRACTIONS: Tuple[float, ...] = (1.00,)  # pot-sized raise only, plus all-in
EPSILON = 1e-9


@dataclass(frozen=True)
class Action:
    kind: str            # "check" | "bet" | "fold" | "call" | "raise" | "allin"
    amount: float = 0.0  # total chips this player puts in THIS action (on top of pending call)

    def __repr__(self) -> str:
        if self.kind in ("check", "fold", "call"):
            return self.kind
        return f"{self.kind}({self.amount:.2f})"


@dataclass
class Node:
    pot: float
    stacks: Tuple[float, float]
    to_call: float
    player_to_act: int
    actions: List[Action] = field(default_factory=list)
    children: List["Node"] = field(default_factory=list)
    is_terminal: bool = False
    terminal_pot: float = 0.0
    terminal_winner: Optional[int] = None  # None => showdown
    depth: int = 0
    # Chance node support (for turn+ solvers). At a chance node, ``children``
    # are indexed in parallel with ``chance_cards`` (card index for each
    # possible run-out). The node is treated as a "pass-through" in CFR —
    # value is averaged uniformly over run-outs. Board-conflict filtering
    # happens downstream in the subtree's showdown table.
    is_chance: bool = False
    chance_cards: List[int] = field(default_factory=list)
    # Showdown terminals may attach a board-specific table identifier so the
    # solver can look up the right outcome matrix. For single-board subgames
    # (river), this is unused; for multi-board subgames (turn), the chance
    # node's children each carry a distinct showdown table via their leaves.
    showdown_board_key: Optional[int] = None


def build_river_tree(
    pot: float,
    stacks: Tuple[float, float],
    first_to_act: int = 0,
    max_raises: int = 4,
) -> Node:
    """Build the full HU river decision tree.

    ``max_raises``: cap on raise depth per chain (defense against pathological
    tiny-raise loops in the abstraction). Most real river trees naturally
    terminate via all-in long before this kicks in.
    """
    if pot < 0:
        raise ValueError("pot must be non-negative")
    if any(s < 0 for s in stacks):
        raise ValueError("stacks must be non-negative")
    if first_to_act not in (0, 1):
        raise ValueError("first_to_act must be 0 or 1")

    return _build_node(
        pot=float(pot),
        stacks=(float(stacks[0]), float(stacks[1])),
        to_call=0.0,
        player_to_act=first_to_act,
        raises_left=max_raises,
        last_aggressor=None,
        depth=0,
    )


def _build_node(
    pot: float,
    stacks: Tuple[float, float],
    to_call: float,
    player_to_act: int,
    raises_left: int,
    last_aggressor: Optional[int],
    depth: int,
) -> Node:
    node = Node(
        pot=pot,
        stacks=stacks,
        to_call=to_call,
        player_to_act=player_to_act,
        depth=depth,
    )

    # Both players all-in before this node: no more actions, showdown.
    if stacks[0] <= EPSILON and stacks[1] <= EPSILON:
        node.is_terminal = True
        node.terminal_pot = pot
        node.terminal_winner = None
        return node

    if to_call <= EPSILON:
        _add_noncall_actions(node, raises_left, last_aggressor)
    else:
        _add_facing_bet_actions(node, raises_left, last_aggressor)

    return node


def _add_noncall_actions(node: Node, raises_left: int, last_aggressor: Optional[int]) -> None:
    me = node.player_to_act
    other = 1 - me
    my_stack = node.stacks[me]

    # Check — passes action to opponent. If opponent already checked this
    # means both have checked and we go to showdown.
    if last_aggressor is None and _both_players_checked_this_round(node, me):
        # Actually we handle this via: if the other player just checked
        # (encoded by last_aggressor=None AND this isn't the root),
        # a second check ends the street at showdown.
        # We handle this in the branch below by making "check" a terminal child.
        pass

    # Build check action
    check_child = _after_check(node, last_aggressor)
    node.actions.append(Action("check"))
    node.children.append(check_child)

    # Build bet actions
    bet_sizes = _bet_candidates(node.pot, my_stack)
    for amount in bet_sizes:
        is_allin = amount >= my_stack - EPSILON
        kind = "allin" if is_allin else "bet"
        node.actions.append(Action(kind, amount))
        node.children.append(
            _after_bet_or_raise(node, amount, is_allin=is_allin, raises_left=raises_left - 1)
        )


def _add_facing_bet_actions(node: Node, raises_left: int, last_aggressor: Optional[int]) -> None:
    me = node.player_to_act
    other = 1 - me
    my_stack = node.stacks[me]
    to_call = min(node.to_call, my_stack)  # can't call more than we have

    # Fold
    fold_child = _after_fold(node)
    node.actions.append(Action("fold"))
    node.children.append(fold_child)

    # Call
    call_child = _after_call(node, to_call)
    node.actions.append(Action("call"))
    node.children.append(call_child)

    # Raise (only if stack allows raising beyond call and budget left)
    if my_stack > to_call + EPSILON and raises_left > 0:
        raise_sizes = _raise_candidates(
            pot_before_call=node.pot,
            to_call=to_call,
            my_stack=my_stack,
        )
        for total_extra in raise_sizes:
            is_allin = total_extra >= my_stack - EPSILON
            kind = "allin" if is_allin else "raise"
            node.actions.append(Action(kind, total_extra))
            node.children.append(
                _after_bet_or_raise(
                    node, total_extra, is_allin=is_allin, raises_left=raises_left - 1
                )
            )


def _bet_candidates(pot: float, stack: float) -> List[float]:
    """Return distinct bet amounts the player can actually make."""
    out: List[float] = []
    seen: List[float] = []
    for frac in BET_FRACTIONS:
        amt = frac * pot
        if amt <= 0 or amt >= stack - EPSILON:
            continue
        if _approx_in(amt, seen):
            continue
        seen.append(amt)
        out.append(amt)
    if stack > 0:
        out.append(stack)  # all-in
    return out


def _raise_candidates(pot_before_call: float, to_call: float, my_stack: float) -> List[float]:
    """Return distinct raise amounts (total extra chips hero puts in this action).

    A 'pot-sized raise' after a bet means: call the bet, then bet the resulting pot.
    total_extra = to_call + (pot_before_call + 2*to_call) * frac
    """
    pot_after_call = pot_before_call + 2 * to_call
    out: List[float] = []
    seen: List[float] = []
    for frac in RAISE_POT_FRACTIONS:
        raise_portion = frac * pot_after_call
        total_extra = to_call + raise_portion
        if total_extra <= to_call + EPSILON or total_extra >= my_stack - EPSILON:
            continue
        if _approx_in(total_extra, seen):
            continue
        seen.append(total_extra)
        out.append(total_extra)
    if my_stack > to_call + EPSILON:
        out.append(my_stack)  # all-in raise
    return out


def _after_check(node: Node, last_aggressor: Optional[int]) -> Node:
    """Handle the check action. If the opponent had the last action and also
    checked (represented by the parent's state), this second check ends the
    street at showdown. We encode 'opponent just checked' by: we're at a node
    where to_call == 0 and player_to_act just switched from the other side."""
    # In this single-street tree, the only way a second check arrives is if the
    # first-to-act checked and now the other player is deciding. When this
    # second player checks, both have checked — showdown.
    # We detect "second check" by: we're at a child of the root checked down,
    # i.e. the parent was first_to_act=other-side with to_call=0 and action=check.
    # Simplest: track whether opener has already checked via a flag. We do this
    # by passing last_aggressor=None through the root, but setting a marker here.
    # Implementation: if last_aggressor is None AND node is not the root in terms
    # of actions, that's a check-check showdown.
    # We encode this by tagging each non-root "to_call=0 & player switched" check
    # as terminal. Practically: if the incoming edge was a check from the other
    # player, this check closes action.
    #
    # We implement the detection by signal: the node we arrived from had
    # to_call == 0 and its action was 'check'. That's exactly the condition
    # that brings us here with last_aggressor == "passed_once".
    # Rather than thread another flag, we observe: in a single-street tree, the
    # only non-opener way to arrive at a to_call=0 node is via a check from the
    # opener. So: if this is the second actor checking, terminate.
    me = node.player_to_act
    other = 1 - me
    # We need to know if the opener has already checked. Since this is the
    # opener's first decision point if depth == 0, otherwise it's the second
    # player responding to a check. We check depth.
    if node.depth == 0:
        # Opener checks — pass action to the other player with to_call still 0
        return _build_node(
            pot=node.pot,
            stacks=node.stacks,
            to_call=0.0,
            player_to_act=other,
            raises_left=0,  # raises don't apply after a check-opened street, but
                             # the other player can still bet fresh; re-enable budget
            last_aggressor=None,
            depth=node.depth + 1,
        )
    # Second check -> showdown
    leaf = Node(
        pot=node.pot,
        stacks=node.stacks,
        to_call=0.0,
        player_to_act=-1,
        is_terminal=True,
        terminal_pot=node.pot,
        terminal_winner=None,
        depth=node.depth + 1,
    )
    return leaf


def _after_bet_or_raise(
    node: Node,
    extra: float,
    is_allin: bool,
    raises_left: int,
) -> Node:
    me = node.player_to_act
    other = 1 - me
    new_stacks = list(node.stacks)
    new_stacks[me] -= extra
    # New pending bet from opponent's POV is the portion above what opponent has
    # already matched. On river single-street, that's simply `extra - to_call_before`.
    # Since before this action to_call was 0 (bet) or something (raise), the new
    # to_call for the opponent is `extra` minus what we were owed — in a bet
    # (no pending call), opponent's to_call is just `extra`. In a raise, we paid
    # to_call first and then raised, so opponent's new to_call is `extra - node.to_call`.
    opp_to_call = extra - node.to_call
    # Opponent can't owe more than they have
    opp_to_call = min(opp_to_call, new_stacks[other])
    new_pot = node.pot + extra + node.to_call  # we committed `extra` plus pre-existing dead money; but node.to_call is what we had to call which is already not in the pot
    # Careful: the bet's `extra` IS the total chips we push into the pot this action.
    # node.to_call was what was owed to match outstanding; if we raise, we paid
    # it as part of extra. So pot grows by extra only.
    new_pot = node.pot + extra

    effective_raises_left = raises_left
    # If either side is all-in after this action, no more raises possible.
    if is_allin or new_stacks[me] <= EPSILON or new_stacks[other] <= EPSILON:
        effective_raises_left = 0

    return _build_node(
        pot=new_pot,
        stacks=(new_stacks[0], new_stacks[1]),
        to_call=opp_to_call,
        player_to_act=other,
        raises_left=effective_raises_left,
        last_aggressor=me,
        depth=node.depth + 1,
    )


def _after_fold(node: Node) -> Node:
    folder = node.player_to_act
    winner = 1 - folder
    return Node(
        pot=node.pot,
        stacks=node.stacks,
        to_call=0.0,
        player_to_act=-1,
        is_terminal=True,
        terminal_pot=node.pot,
        terminal_winner=winner,
        depth=node.depth + 1,
    )


def _after_call(node: Node, to_call: float) -> Node:
    me = node.player_to_act
    other = 1 - me
    new_stacks = list(node.stacks)
    new_stacks[me] -= to_call
    new_pot = node.pot + to_call
    # Call closes the action on this street — always terminal (showdown) on river.
    return Node(
        pot=new_pot,
        stacks=(new_stacks[0], new_stacks[1]),
        to_call=0.0,
        player_to_act=-1,
        is_terminal=True,
        terminal_pot=new_pot,
        terminal_winner=None,
        depth=node.depth + 1,
    )


def _approx_in(x: float, lst: List[float]) -> bool:
    return any(abs(x - y) < EPSILON for y in lst)


def _both_players_checked_this_round(node: Node, me: int) -> bool:
    return False


def walk_nodes(root: Node):
    """Iterate the tree in DFS order, yielding each node."""
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        if not n.is_terminal:
            stack.extend(n.children)


def count_nodes(root: Node) -> int:
    return sum(1 for _ in walk_nodes(root))


def count_terminals(root: Node) -> int:
    return sum(1 for n in walk_nodes(root) if n.is_terminal)


def count_decision_nodes(root: Node) -> int:
    return sum(1 for n in walk_nodes(root) if not n.is_terminal and not n.is_chance)


def count_chance_nodes(root: Node) -> int:
    return sum(1 for n in walk_nodes(root) if n.is_chance)


def build_turn_tree(
    board_4,
    pot: float,
    stacks: Tuple[float, float],
    first_to_act: int = 0,
    max_raises: int = 2,
    river_max_raises: int = 2,
) -> Node:
    """Build the turn HU subgame tree.

    Structure: turn decisions (same action space as river) → chance node
    (river card dealt uniformly from remaining 48) → river decision subtree.

    Each chance node has one child per remaining card. Each child is a full
    river decision tree. Showdown terminals within each river subtree carry a
    ``showdown_board_key`` pointing to the 5-card board those terminals use
    (needed by the solver to pick the right outcome matrix).

    Fold terminals on the turn keep their standard ``terminal_winner`` semantics
    and are NOT replaced with chance nodes (the hand ends without a river).
    """
    from .cards import card_to_index

    if len(board_4) != 4:
        raise ValueError("board_4 must have exactly 4 cards")
    board_indices = {card_to_index(c) for c in board_4}
    if len(board_indices) != 4:
        raise ValueError("board_4 has duplicate cards")
    remaining_cards = [c for c in range(52) if c not in board_indices]

    # Build the turn action tree reusing the river-tree builder (which doesn't
    # touch board cards — it only shapes the action space).
    turn_tree = build_river_tree(
        pot=pot, stacks=stacks, first_to_act=first_to_act, max_raises=max_raises
    )

    # Walk the tree and replace showdown terminals with chance nodes.
    _expand_showdowns_to_chance(turn_tree, remaining_cards, river_max_raises)
    return turn_tree


def _expand_showdowns_to_chance(
    node: Node, remaining_cards: List[int], river_max_raises: int
) -> None:
    """In-place: at every showdown terminal in ``node``'s subtree, replace it
    with a chance node whose children are full river decision trees.

    Fold terminals are untouched. Forced double-all-in showdowns become chance
    nodes whose river subtrees are trivial (just a showdown — build_river_tree
    emits an immediate terminal when both stacks are 0)."""
    if node.is_terminal and node.terminal_winner is None:
        # Showdown endpoint — convert to chance node
        pot_at_chance = node.terminal_pot
        stacks_at_chance = node.stacks
        # River first-to-act is always player 0 by convention.
        children: List[Node] = []
        for card_idx in remaining_cards:
            river_root = build_river_tree(
                pot=pot_at_chance,
                stacks=stacks_at_chance,
                first_to_act=0,
                max_raises=river_max_raises,
            )
            children.append(river_root)
        node.is_terminal = False
        node.is_chance = True
        node.chance_cards = list(remaining_cards)
        node.children = children
        # Walk each child river subtree and tag its showdown terminals with
        # the corresponding 5-card-board key (we'll use (board_4_hash, card) later)
        for card_idx, river_root in zip(remaining_cards, children):
            _tag_river_showdowns(river_root, card_idx)
        return
    for child in node.children:
        _expand_showdowns_to_chance(child, remaining_cards, river_max_raises)


def _tag_river_showdowns(node: Node, river_card_idx: int) -> None:
    if node.is_terminal and node.terminal_winner is None:
        node.showdown_board_key = river_card_idx
        return
    if node.is_chance:
        for child in node.children:
            _tag_river_showdowns(child, river_card_idx)
        return
    for child in node.children:
        _tag_river_showdowns(child, river_card_idx)


def build_flop_tree(
    board_3,
    pot: float,
    stacks: Tuple[float, float],
    first_to_act: int = 0,
    max_raises: int = 1,
    turn_max_raises: int = 1,
    river_max_raises: int = 1,
) -> Node:
    """Build the flop HU subgame tree.

    Structure: flop decisions → chance (turn card, 49 options) → turn subtree
    (which itself contains chance → river subtree → showdown).

    Showdown leaves at the deepest level carry ``showdown_board_key`` as a
    ``(turn_card_idx, river_card_idx)`` tuple, so the solver can look up the
    appropriate per-(turn,river) outcome matrix.
    """
    from .cards import card_to_index, INDEX_TO_CARD

    if len(board_3) != 3:
        raise ValueError("board_3 must have exactly 3 cards")
    board_indices = {card_to_index(c) for c in board_3}
    if len(board_indices) != 3:
        raise ValueError("board_3 has duplicate cards")
    remaining_cards = [c for c in range(52) if c not in board_indices]

    # Build flop action tree (reuse river-tree builder, agnostic of cards).
    flop_tree = build_river_tree(
        pot=pot, stacks=stacks, first_to_act=first_to_act, max_raises=max_raises
    )

    # Replace each showdown terminal with a chance(turn) → turn subtree.
    _expand_flop_showdowns(
        flop_tree,
        list(board_3),
        remaining_cards,
        turn_max_raises,
        river_max_raises,
    )
    return flop_tree


def _expand_flop_showdowns(
    node: Node,
    board_3: List[str],
    remaining_cards: List[int],
    turn_max_raises: int,
    river_max_raises: int,
) -> None:
    from .cards import INDEX_TO_CARD

    if node.is_terminal and node.terminal_winner is None:
        # Showdown endpoint on flop → chance(turn) + 49 turn subtrees
        children: List[Node] = []
        for turn_card_idx in remaining_cards:
            turn_card_str = INDEX_TO_CARD[turn_card_idx]
            new_board_4 = list(board_3) + [turn_card_str]
            turn_root = build_turn_tree(
                board_4=new_board_4,
                pot=node.terminal_pot,
                stacks=node.stacks,
                first_to_act=0,  # post-turn action: OOP first by convention
                max_raises=turn_max_raises,
                river_max_raises=river_max_raises,
            )
            children.append(turn_root)
        node.is_terminal = False
        node.is_chance = True
        node.chance_cards = list(remaining_cards)
        node.children = children
        # Re-tag showdown_board_key on the deepest leaves: combine the turn
        # card we just dealt with the river card already tagged by build_turn_tree.
        for tc, child in zip(remaining_cards, children):
            _retag_with_turn_card(child, tc)
        return
    for child in node.children:
        _expand_flop_showdowns(
            child, board_3, remaining_cards, turn_max_raises, river_max_raises
        )


def _retag_with_turn_card(node: Node, turn_card_idx: int) -> None:
    """Walk a turn subtree's deepest showdowns; replace river-int key with
    a (turn_card, river_card) tuple key."""
    if node.is_terminal and node.terminal_winner is None:
        if isinstance(node.showdown_board_key, int):
            node.showdown_board_key = (turn_card_idx, node.showdown_board_key)
        return
    for child in node.children:
        _retag_with_turn_card(child, turn_card_idx)

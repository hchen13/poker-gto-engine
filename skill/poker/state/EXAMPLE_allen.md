# Player: Allen

- **Stakes**: ¥1/¥2 cash (200BB buy-in = ¥400)
- **Type**: huge fish (~2000+ above water early in this session)
- **Specific leaks**:
  - Limps UTG, then 3-bets when there's an iso-raise behind. Range is heavily concentrated on big pairs (AA/KK/QQ/JJ) plus AK; almost never balances with bluffs.
  - Cbets flop ~33% pot with full continuing range; doesn't slow down with overpairs on coordinated boards.
  - Turn aggression is mostly value-driven; rarely turn-bluffs after a passive line.
  - Will jam turn with overpairs facing a raise (over-values single-pair holdings vs. set/two-pair raises).

- **Hand history references**:
  - 2026-04-19 1/2 game: 99 in CO vs Allen UTG limp-3bet. Flop KJ9 rainbow, turn 5 bet 100, hero raised 300, Allen jammed. (See `fixtures/rust/turn_99_vs_fish.json` for the solver setup.)

- **Exploitative adjustments**:
  - When Allen takes the limp-3bet line: assume value-only range. Bluff-catching with TPTK+ is profitable; sets/two-pairs CALL turn don't raise (raising folds out the overpairs we beat).
  - When Allen jams turn: range = nuts (KK/JJ on KJ9-x boards) + overpair "I have a pair" frustration jams. Pot odds of <35% to call → call wide.

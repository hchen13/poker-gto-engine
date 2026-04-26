# Archetype: Reg

Studied regular; plays close-to-GTO preflop with some exploitative postflop leaks.

- **Range tendencies**:
  - Preflop: near-solver (~20% UTG, 28% HJ, 35% CO, 45% BTN, varying 3bet/4bet freq)
  - Has bluff 3-bets (A5s, A4s, KJo occasionally from SB); balanced 4-bet range
  - Flats wider from BTN (22+, AT+, KTs+, QTs+, JTs, T9s, 98s, 87s)

- **Postflop tendencies**:
  - Cbets flop at mostly-GTO frequency (~60-70% on dry boards, ~45% on wet)
  - Barrels turn with correct polarized range (nuts + double-barrel draws)
  - Rivers are closer to GTO; bluff frequencies approximately right
  - May over-fold vs large river overbets (common reg leak)
  - Occasionally over-bluffs turn when OOP caller checks back flop

- **Exploitative adjustments**:
  - Play closer to GTO output; reg is hardest to exploit
  - Overbet rivers vs reg for fold equity (they over-fold)
  - Check-call more vs reg turn bets (they over-bluff turn in checked-to spots)
  - Don't try to bluff them off strong hands; their calling ranges are correct

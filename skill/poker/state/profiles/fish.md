# Archetype: Fish

Default profile for loose-passive recreational players.

- **Range tendencies**:
  - Preflop: very wide opening + calling range (50%+ from most positions)
  - 3-bet range is value-heavy (JJ+, AKs/o); rarely 3-bet bluffs
  - Calls 3-bets with wide speculative range (any suited A, broadway offsuits)

- **Postflop tendencies**:
  - Bets flops with continuing range (~40-60% cbet)
  - Rarely barrels turn as bluff; most turn bets are value
  - Overvalues single-pair hands; river bet sizes don't correlate well with strength
  - Doesn't bluff-raise rivers
  - Calls down thin; hard to bluff off made hands

- **Sizing tells**:
  - Small bet ≈ weak/draw; large bet ≈ nuts or overpair
  - Pot-sized+ river bets are heavily value-weighted

- **Exploitative adjustments** (LLM applies these as 理由 anchors; solver output not modified):
  - Thin value bet wider than GTO on rivers
  - Don't bluff rivers — they call too wide
  - Raise flops/turns with strong made hands for value; don't balance with bluffs
  - Fold middle-pair / bluff-catchers to big river bets (their ratio is nuts-heavy)

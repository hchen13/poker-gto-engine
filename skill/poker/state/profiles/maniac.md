# Archetype: Maniac

Loose-aggressive player with high 3-bet/4-bet/cbet frequencies. Often unbalanced bluff-heavy.

- **Range tendencies**:
  - Preflop: very wide opens (~60%+ BTN); 3-bets 15%+ (GTO ~8-10%)
  - 3-bet and 4-bet ranges heavily bluff-skewed (lots of suited connectors, low suited Ax)
  - Calls 3-bets lighter than GTO

- **Postflop tendencies**:
  - Cbets nearly every flop (range bet). Barrels turn often (bluff-heavy)
  - Will raise flops as bluff with naked overcard + backdoor draws
  - River: polarized but bluff-frequency elevated. Bet sizing mixed
  - Rarely check-back strong hands — they bet everything

- **Exploitative adjustments**:
  - Call wider vs their aggression (their bluff-to-value ratio is too high)
  - Don't fold top pair to single barrels; often the right call into river
  - 3-bet them wider for value (their call-3bet range is wide-weak)
  - When they slow down (check turn after cbet flop), they usually have air — probe into them
  - River hero-call lighter; solver minimum defense frequency is a floor, not a ceiling vs maniac

use poker_gto_engine::kuhn::{train_kuhn_cfr, KuhnTrainingSummary};

fn assert_summary(summary: KuhnTrainingSummary) {
    assert!((summary.player_0_value + (1.0 / 18.0)).abs() < 0.02);

    let jack_bet = summary.root_strategy.get("J").unwrap().bet;
    let queen_bet = summary.root_strategy.get("Q").unwrap().bet;
    let king_bet = summary.root_strategy.get("K").unwrap().bet;

    assert!(jack_bet >= 0.0);
    assert!(jack_bet <= (1.0 / 3.0) + 0.03);
    assert!((king_bet - f64::min(3.0 * jack_bet, 1.0)).abs() < 0.08);
    assert!(queen_bet.abs() < 0.05);
}

#[test]
fn rust_kuhn_cfr_converges_to_known_equilibrium_family() {
    let summary = train_kuhn_cfr(20_000);
    assert_summary(summary);
}

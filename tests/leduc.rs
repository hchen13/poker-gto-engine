use poker_gto_engine::leduc::{train_leduc_cfr, LeducTrainingSummary};

fn assert_reference_shape(summary: &LeducTrainingSummary) {
    assert!((summary.player_0_value - (-0.08553)).abs() < 0.03);

    let jack_bet = summary.root_strategy.get("J").unwrap().bet;
    let queen_bet = summary.root_strategy.get("Q").unwrap().bet;
    let king_bet = summary.root_strategy.get("K").unwrap().bet;

    assert!(jack_bet < 0.2);
    assert!(queen_bet > 0.6);
    assert!(king_bet > 0.6);
}

#[test]
fn rust_leduc_cfr_matches_reference_shape() {
    let summary = train_leduc_cfr(100);
    assert_reference_shape(&summary);
}

#[test]
fn rust_leduc_exploitability_improves_with_more_iterations() {
    let shallow = train_leduc_cfr(100);
    let deeper = train_leduc_cfr(1_000);

    assert!(shallow.exploitability >= 0.0);
    assert!(deeper.exploitability >= 0.0);
    assert!(deeper.exploitability < shallow.exploitability);
    assert!(deeper.exploitability < 0.05);
}

use poker_gto_engine::leduc::train_leduc_cfr;
use std::env;

fn main() {
    let iterations = env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);

    let summary = train_leduc_cfr(iterations);
    println!(
        "{{\"player_0_value\":{:.12},\"training_game_value\":{:.12},\"exploitability\":{:.12},\"best_response_player_0\":{:.12},\"best_response_player_1\":{:.12},\"infoset_count\":{},\"root_strategy\":{{\"J\":{{\"check\":{:.12},\"bet\":{:.12}}},\"Q\":{{\"check\":{:.12},\"bet\":{:.12}}},\"K\":{{\"check\":{:.12},\"bet\":{:.12}}}}}}}",
        summary.player_0_value,
        summary.training_game_value,
        summary.exploitability,
        summary.best_response_player_0,
        summary.best_response_player_1,
        summary.infoset_count,
        summary.root_strategy["J"].check,
        summary.root_strategy["J"].bet,
        summary.root_strategy["Q"].check,
        summary.root_strategy["Q"].bet,
        summary.root_strategy["K"].check,
        summary.root_strategy["K"].bet,
    );
}

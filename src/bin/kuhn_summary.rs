use poker_gto_engine::kuhn::train_kuhn_cfr;
use std::env;

fn main() {
    let iterations = env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20_000);

    let summary = train_kuhn_cfr(iterations);
    println!(
        "{{\"player_0_value\":{:.12},\"root_strategy\":{{\"J\":{{\"check\":{:.12},\"bet\":{:.12}}},\"Q\":{{\"check\":{:.12},\"bet\":{:.12}}},\"K\":{{\"check\":{:.12},\"bet\":{:.12}}}}}}}",
        summary.player_0_value,
        summary.root_strategy["J"].check,
        summary.root_strategy["J"].bet,
        summary.root_strategy["Q"].check,
        summary.root_strategy["Q"].bet,
        summary.root_strategy["K"].check,
        summary.root_strategy["K"].bet,
    );
}

use std::collections::BTreeMap;

const PASS_ACTION: char = 'p';
const BET_ACTION: char = 'b';
const CARDS: [char; 3] = ['J', 'Q', 'K'];

#[derive(Debug, Clone, Copy)]
pub struct ActionMix {
    pub check: f64,
    pub bet: f64,
}

#[derive(Debug, Clone)]
pub struct KuhnTrainingSummary {
    pub player_0_value: f64,
    pub root_strategy: BTreeMap<String, ActionMix>,
}

#[derive(Default)]
struct InfoSet {
    regret_sum: [f64; 2],
    strategy_sum: [f64; 2],
}

impl InfoSet {
    fn current_strategy(&self) -> [f64; 2] {
        let positive_regrets = [self.regret_sum[0].max(0.0), self.regret_sum[1].max(0.0)];
        let normalizer = positive_regrets[0] + positive_regrets[1];

        if normalizer > 0.0 {
            [positive_regrets[0] / normalizer, positive_regrets[1] / normalizer]
        } else {
            [0.5, 0.5]
        }
    }

    fn average_strategy(&self) -> ActionMix {
        let normalizer = self.strategy_sum[0] + self.strategy_sum[1];
        if normalizer > 0.0 {
            ActionMix {
                check: self.strategy_sum[0] / normalizer,
                bet: self.strategy_sum[1] / normalizer,
            }
        } else {
            ActionMix { check: 0.5, bet: 0.5 }
        }
    }
}

#[derive(Default)]
struct KuhnTrainer {
    info_sets: BTreeMap<String, InfoSet>,
}

impl KuhnTrainer {
    fn train(&mut self, iterations: usize) -> KuhnTrainingSummary {
        let mut utility_sum = 0.0;
        let permutations = [
            ['J', 'Q'],
            ['J', 'K'],
            ['Q', 'J'],
            ['Q', 'K'],
            ['K', 'J'],
            ['K', 'Q'],
        ];

        for _ in 0..iterations {
            for cards in permutations {
                utility_sum += self.cfr(cards, String::new(), 1.0, 1.0);
            }
        }

        let mut root_strategy = BTreeMap::new();
        for card in CARDS {
            if let Some(info_set) = self.info_sets.get(&card.to_string()) {
                root_strategy.insert(card.to_string(), info_set.average_strategy());
            }
        }

        KuhnTrainingSummary {
            player_0_value: utility_sum / (iterations as f64 * 6.0),
            root_strategy,
        }
    }

    fn cfr(&mut self, cards: [char; 2], history: String, reach_0: f64, reach_1: f64) -> f64 {
        if let Some(terminal) = terminal_utility(cards, &history) {
            return terminal;
        }

        let player = history.len() % 2;
        let info_key = format!("{}{}", cards[player], history);
        let strategy = {
            let info_set = self.info_sets.entry(info_key.clone()).or_default();
            let strategy = info_set.current_strategy();
            if player == 0 {
                info_set.strategy_sum[0] += reach_0 * strategy[0];
                info_set.strategy_sum[1] += reach_0 * strategy[1];
            } else {
                info_set.strategy_sum[0] += reach_1 * strategy[0];
                info_set.strategy_sum[1] += reach_1 * strategy[1];
            }
            strategy
        };

        let mut action_utilities = [0.0, 0.0];
        let mut node_utility = 0.0;

        for (index, action) in [PASS_ACTION, BET_ACTION].iter().enumerate() {
            let mut next_history = history.clone();
            next_history.push(*action);
            action_utilities[index] = if player == 0 {
                -self.cfr(cards, next_history, reach_0 * strategy[index], reach_1)
            } else {
                -self.cfr(cards, next_history, reach_0, reach_1 * strategy[index])
            };
            node_utility += strategy[index] * action_utilities[index];
        }

        let info_set = self.info_sets.get_mut(&info_key).unwrap();
        for index in 0..2 {
            let regret = action_utilities[index] - node_utility;
            if player == 0 {
                info_set.regret_sum[index] += reach_1 * regret;
            } else {
                info_set.regret_sum[index] += reach_0 * regret;
            }
        }

        node_utility
    }
}

fn terminal_utility(cards: [char; 2], history: &str) -> Option<f64> {
    if history.len() < 2 {
        return None;
    }

    let player = history.len() % 2;
    let opponent = 1 - player;
    let bytes = history.as_bytes();
    let terminal_pass = *bytes.last().unwrap() == PASS_ACTION as u8;
    let double_bet = bytes[history.len() - 2] == BET_ACTION as u8 && bytes[history.len() - 1] == BET_ACTION as u8;

    if terminal_pass {
        if history == "pp" {
            return Some(if card_rank(cards[player]) > card_rank(cards[opponent]) { 1.0 } else { -1.0 });
        }
        return Some(1.0);
    }

    if double_bet {
        return Some(if card_rank(cards[player]) > card_rank(cards[opponent]) { 2.0 } else { -2.0 });
    }

    None
}

fn card_rank(card: char) -> usize {
    match card {
        'J' => 0,
        'Q' => 1,
        'K' => 2,
        _ => panic!("unexpected card"),
    }
}

pub fn train_kuhn_cfr(iterations: usize) -> KuhnTrainingSummary {
    let mut trainer = KuhnTrainer::default();
    trainer.train(iterations)
}

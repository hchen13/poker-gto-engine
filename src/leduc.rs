use std::collections::BTreeMap;

const RANKS: [char; 3] = ['J', 'Q', 'K'];
const DECK: [&str; 6] = ["J1", "J2", "Q1", "Q2", "K1", "K2"];
const INITIAL_ANTE: i32 = 1;
const BET_SIZES: [i32; 2] = [2, 4];
const MAX_RAISES_PER_ROUND: u8 = 2;

#[derive(Debug, Clone, Copy)]
pub struct ActionMix {
    pub check: f64,
    pub bet: f64,
}

#[derive(Debug, Clone)]
pub struct LeducTrainingSummary {
    pub player_0_value: f64,
    pub training_game_value: f64,
    pub root_strategy: BTreeMap<String, ActionMix>,
    pub infoset_count: usize,
    pub best_response_player_0: f64,
    pub best_response_player_1: f64,
    pub exploitability: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Action {
    Check,
    Bet,
    Fold,
    Call,
    Raise,
}

impl Action {
    fn history_char(&self) -> char {
        match self {
            Action::Check => 'x',
            Action::Bet => 'b',
            Action::Fold => 'f',
            Action::Call => 'c',
            Action::Raise => 'r',
        }
    }
}

#[derive(Debug, Clone)]
struct StrategySummary {
    actions: Vec<Action>,
    probabilities: Vec<f64>,
}

impl StrategySummary {
    fn action_probability(&self, action: &Action) -> f64 {
        self.actions
            .iter()
            .zip(self.probabilities.iter())
            .find(|(candidate, _)| *candidate == action)
            .map(|(_, probability)| *probability)
            .unwrap_or(0.0)
    }
}

#[derive(Debug, Clone)]
struct InfoSet {
    actions: Vec<Action>,
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
}

impl InfoSet {
    fn new(actions: &[Action]) -> Self {
        Self {
            actions: actions.to_vec(),
            regret_sum: vec![0.0; actions.len()],
            strategy_sum: vec![0.0; actions.len()],
        }
    }

    fn current_strategy(&self) -> Vec<f64> {
        let positive_regrets: Vec<f64> = self.regret_sum.iter().map(|regret| regret.max(0.0)).collect();
        let normalizer: f64 = positive_regrets.iter().sum();
        if normalizer > 0.0 {
            positive_regrets.iter().map(|value| value / normalizer).collect()
        } else {
            vec![1.0 / self.actions.len() as f64; self.actions.len()]
        }
    }

    fn average_strategy(&self) -> StrategySummary {
        let normalizer: f64 = self.strategy_sum.iter().sum();
        let probabilities = if normalizer > 0.0 {
            self.strategy_sum.iter().map(|value| value / normalizer).collect()
        } else {
            vec![1.0 / self.actions.len() as f64; self.actions.len()]
        };
        StrategySummary {
            actions: self.actions.clone(),
            probabilities,
        }
    }
}

#[derive(Debug, Clone)]
struct BestResponseInfoSetStats {
    value_sum: Vec<f64>,
    weight_sum: f64,
}

impl BestResponseInfoSetStats {
    fn new(action_count: usize) -> Self {
        Self {
            value_sum: vec![0.0; action_count],
            weight_sum: 0.0,
        }
    }

    fn accumulate(&mut self, opponent_reach: f64, action_values: &[f64]) {
        self.weight_sum += opponent_reach;
        for (index, value) in action_values.iter().enumerate() {
            self.value_sum[index] += opponent_reach * value;
        }
    }

    fn best_action_index(&self) -> usize {
        if self.weight_sum == 0.0 {
            return 0;
        }
        let mut best_index = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (index, total) in self.value_sum.iter().enumerate() {
            let average_value = total / self.weight_sum;
            if average_value > best_value {
                best_value = average_value;
                best_index = index;
            }
        }
        best_index
    }
}

#[derive(Debug, Clone)]
struct LeducState {
    private_cards: [String; 2],
    public_card: Option<String>,
    round_index: usize,
    current_player: usize,
    contributions: [i32; 2],
    round_contributions: [i32; 2],
    raises_in_round: u8,
    round_histories: [String; 2],
    folded_player: Option<usize>,
}

impl LeducState {
    fn initial(private_cards: [&str; 2]) -> Self {
        Self {
            private_cards: [private_cards[0].to_string(), private_cards[1].to_string()],
            public_card: None,
            round_index: 0,
            current_player: 0,
            contributions: [INITIAL_ANTE, INITIAL_ANTE],
            round_contributions: [0, 0],
            raises_in_round: 0,
            round_histories: [String::new(), String::new()],
            folded_player: None,
        }
    }

    fn is_chance_pending(&self) -> bool {
        self.round_index == 1 && self.public_card.is_none() && self.folded_player.is_none()
    }
}

#[derive(Default)]
struct LeducTrainer {
    info_sets: BTreeMap<String, InfoSet>,
}

impl LeducTrainer {
    fn train(&mut self, iterations: usize) -> LeducTrainingSummary {
        let private_deals = private_deals();
        let mut utility_sum = 0.0;

        for _ in 0..iterations {
            for private_cards in &private_deals {
                utility_sum += self.cfr(LeducState::initial(*private_cards), 1.0, 1.0);
            }
        }

        let training_game_value = utility_sum / (iterations as f64 * private_deals.len() as f64);
        let infoset_strategy: BTreeMap<String, StrategySummary> = self
            .info_sets
            .iter()
            .map(|(key, info_set)| (key.clone(), info_set.average_strategy()))
            .collect();

        let mut root_strategy = BTreeMap::new();
        for rank in RANKS {
            let key = infoset_key(rank, None, [&String::new(), &String::new()]);
            if let Some(strategy) = infoset_strategy.get(&key) {
                root_strategy.insert(
                    rank.to_string(),
                    ActionMix {
                        check: strategy.action_probability(&Action::Check),
                        bet: strategy.action_probability(&Action::Bet),
                    },
                );
            }
        }

        let average_strategy_value = self.evaluate_average_strategy(&infoset_strategy, &private_deals);
        let best_response_player_0 = self.best_response_value(&infoset_strategy, 0, &private_deals);
        let best_response_player_1 = self.best_response_value(&infoset_strategy, 1, &private_deals);
        let exploitability =
            ((best_response_player_0 - average_strategy_value) + (best_response_player_1 - (-average_strategy_value))) / 2.0;

        LeducTrainingSummary {
            player_0_value: average_strategy_value,
            training_game_value,
            root_strategy,
            infoset_count: self.info_sets.len(),
            best_response_player_0,
            best_response_player_1,
            exploitability,
        }
    }

    fn cfr(&mut self, state: LeducState, reach_0: f64, reach_1: f64) -> f64 {
        if let Some(terminal) = terminal_utility(&state) {
            return terminal;
        }

        if state.is_chance_pending() {
            let remaining_cards = remaining_public_cards(&state);
            let chance_weight = 1.0 / remaining_cards.len() as f64;
            return remaining_cards
                .iter()
                .map(|public_card| chance_weight * self.cfr(deal_public_card(&state, public_card), reach_0, reach_1))
                .sum();
        }

        let actions = legal_actions(&state);
        let infoset_key = state_infoset_key(&state);
        let player = state.current_player;
        let strategy = {
            let info_set = self
                .info_sets
                .entry(infoset_key.clone())
                .or_insert_with(|| InfoSet::new(&actions));
            let strategy = info_set.current_strategy();
            let reach = if player == 0 { reach_0 } else { reach_1 };
            for (index, probability) in strategy.iter().enumerate() {
                info_set.strategy_sum[index] += reach * probability;
            }
            strategy
        };

        let mut action_utilities = vec![0.0; actions.len()];
        let mut node_utility = 0.0;
        for (index, action) in actions.iter().enumerate() {
            let next_state = apply_action(&state, action);
            action_utilities[index] = if player == 0 {
                self.cfr(next_state, reach_0 * strategy[index], reach_1)
            } else {
                self.cfr(next_state, reach_0, reach_1 * strategy[index])
            };
            node_utility += strategy[index] * action_utilities[index];
        }

        let info_set = self.info_sets.get_mut(&infoset_key).unwrap();
        for index in 0..actions.len() {
            if player == 0 {
                info_set.regret_sum[index] += reach_1 * (action_utilities[index] - node_utility);
            } else {
                info_set.regret_sum[index] += reach_0 * (node_utility - action_utilities[index]);
            }
        }

        node_utility
    }

    fn evaluate_average_strategy(
        &self,
        infoset_strategy: &BTreeMap<String, StrategySummary>,
        private_deals: &Vec<[&'static str; 2]>,
    ) -> f64 {
        let total: f64 = private_deals
            .iter()
            .map(|private_cards| evaluate_strategy_profile(&LeducState::initial(*private_cards), infoset_strategy))
            .sum();
        total / private_deals.len() as f64
    }

    fn best_response_value(
        &self,
        infoset_strategy: &BTreeMap<String, StrategySummary>,
        br_player: usize,
        private_deals: &Vec<[&'static str; 2]>,
    ) -> f64 {
        let mut policy: BTreeMap<String, usize> = BTreeMap::new();

        for _ in 0..10 {
            let mut stats: BTreeMap<String, BestResponseInfoSetStats> = BTreeMap::new();
            for private_cards in private_deals {
                evaluate_with_policy(
                    &LeducState::initial(*private_cards),
                    infoset_strategy,
                    br_player,
                    1.0,
                    &mut policy,
                    &mut stats,
                );
            }

            let mut changed = false;
            for (key, info_stats) in stats.iter() {
                let best_index = info_stats.best_action_index();
                if policy.get(key) != Some(&best_index) {
                    policy.insert(key.clone(), best_index);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let total: f64 = private_deals
            .iter()
            .map(|private_cards| evaluate_final_policy(&LeducState::initial(*private_cards), infoset_strategy, br_player, &policy))
            .sum();
        total / private_deals.len() as f64
    }
}

fn evaluate_strategy_profile(state: &LeducState, infoset_strategy: &BTreeMap<String, StrategySummary>) -> f64 {
    if let Some(terminal) = terminal_utility(state) {
        return terminal;
    }

    if state.is_chance_pending() {
        let remaining_cards = remaining_public_cards(state);
        let chance_weight = 1.0 / remaining_cards.len() as f64;
        return remaining_cards
            .iter()
            .map(|public_card| chance_weight * evaluate_strategy_profile(&deal_public_card(state, public_card), infoset_strategy))
            .sum();
    }

    let actions = legal_actions(state);
    let strategy = strategy_for_state(state, &actions, infoset_strategy);
    actions
        .iter()
        .enumerate()
        .map(|(index, action)| strategy[index] * evaluate_strategy_profile(&apply_action(state, action), infoset_strategy))
        .sum()
}

fn evaluate_with_policy(
    state: &LeducState,
    infoset_strategy: &BTreeMap<String, StrategySummary>,
    br_player: usize,
    opponent_reach: f64,
    policy: &mut BTreeMap<String, usize>,
    stats: &mut BTreeMap<String, BestResponseInfoSetStats>,
) -> f64 {
    if let Some(terminal) = terminal_utility(state) {
        return if br_player == 0 { terminal } else { -terminal };
    }

    if state.is_chance_pending() {
        let remaining_cards = remaining_public_cards(state);
        let chance_weight = 1.0 / remaining_cards.len() as f64;
        return remaining_cards
            .iter()
            .map(|public_card| {
                chance_weight
                    * evaluate_with_policy(
                        &deal_public_card(state, public_card),
                        infoset_strategy,
                        br_player,
                        opponent_reach * chance_weight,
                        policy,
                        stats,
                    )
            })
            .sum();
    }

    let actions = legal_actions(state);
    if state.current_player == br_player {
        let action_values: Vec<f64> = actions
            .iter()
            .map(|action| evaluate_with_policy(&apply_action(state, action), infoset_strategy, br_player, opponent_reach, policy, stats))
            .collect();
        let key = state_infoset_key(state);
        stats
            .entry(key.clone())
            .or_insert_with(|| BestResponseInfoSetStats::new(actions.len()))
            .accumulate(opponent_reach, &action_values);
        let policy_index = *policy.entry(key).or_insert(0);
        action_values[policy_index]
    } else {
        let strategy = strategy_for_state(state, &actions, infoset_strategy);
        actions
            .iter()
            .enumerate()
            .map(|(index, action)| {
                strategy[index]
                    * evaluate_with_policy(
                        &apply_action(state, action),
                        infoset_strategy,
                        br_player,
                        opponent_reach * strategy[index],
                        policy,
                        stats,
                    )
            })
            .sum()
    }
}

fn evaluate_final_policy(
    state: &LeducState,
    infoset_strategy: &BTreeMap<String, StrategySummary>,
    br_player: usize,
    policy: &BTreeMap<String, usize>,
) -> f64 {
    if let Some(terminal) = terminal_utility(state) {
        return if br_player == 0 { terminal } else { -terminal };
    }

    if state.is_chance_pending() {
        let remaining_cards = remaining_public_cards(state);
        let chance_weight = 1.0 / remaining_cards.len() as f64;
        return remaining_cards
            .iter()
            .map(|public_card| chance_weight * evaluate_final_policy(&deal_public_card(state, public_card), infoset_strategy, br_player, policy))
            .sum();
    }

    let actions = legal_actions(state);
    if state.current_player == br_player {
        let key = state_infoset_key(state);
        let action_index = *policy.get(&key).unwrap();
        evaluate_final_policy(&apply_action(state, &actions[action_index]), infoset_strategy, br_player, policy)
    } else {
        let strategy = strategy_for_state(state, &actions, infoset_strategy);
        actions
            .iter()
            .enumerate()
            .map(|(index, action)| {
                strategy[index] * evaluate_final_policy(&apply_action(state, action), infoset_strategy, br_player, policy)
            })
            .sum()
    }
}

fn strategy_for_state(
    state: &LeducState,
    actions: &[Action],
    infoset_strategy: &BTreeMap<String, StrategySummary>,
) -> Vec<f64> {
    let key = state_infoset_key(state);
    if let Some(strategy) = infoset_strategy.get(&key) {
        actions.iter().map(|action| strategy.action_probability(action)).collect()
    } else {
        vec![1.0 / actions.len() as f64; actions.len()]
    }
}

fn legal_actions(state: &LeducState) -> Vec<Action> {
    let opponent = 1 - state.current_player;
    let outstanding = state.round_contributions[state.current_player] < state.round_contributions[opponent];
    if outstanding {
        let mut actions = vec![Action::Fold, Action::Call];
        if state.raises_in_round < MAX_RAISES_PER_ROUND {
            actions.push(Action::Raise);
        }
        actions
    } else {
        vec![Action::Check, Action::Bet]
    }
}

fn apply_action(state: &LeducState, action: &Action) -> LeducState {
    if state.is_chance_pending() {
        panic!("cannot apply player action while public card is pending");
    }

    let mut next = state.clone();
    let player = state.current_player;
    let opponent = 1 - player;
    next.round_histories[state.round_index].push(action.history_char());
    let bet_size = BET_SIZES[state.round_index];

    match action {
        Action::Fold => {
            next.folded_player = Some(player);
        }
        Action::Check => {
            if state.round_histories[state.round_index] == "x" {
                next = advance_round_or_terminal(&next);
            } else {
                next.current_player = opponent;
            }
        }
        Action::Bet => {
            next.contributions[player] += bet_size;
            next.round_contributions[player] += bet_size;
            next.current_player = opponent;
            next.raises_in_round = 1;
        }
        Action::Call => {
            let call_amount = state.round_contributions[opponent] - state.round_contributions[player];
            next.contributions[player] += call_amount;
            next.round_contributions[player] += call_amount;
            next = advance_round_or_terminal(&next);
        }
        Action::Raise => {
            let raise_amount = (state.round_contributions[opponent] - state.round_contributions[player]) + bet_size;
            next.contributions[player] += raise_amount;
            next.round_contributions[player] += raise_amount;
            next.current_player = opponent;
            next.raises_in_round += 1;
        }
    }

    next
}

fn deal_public_card(state: &LeducState, public_card: &str) -> LeducState {
    let mut next = state.clone();
    next.public_card = Some(public_card.to_string());
    next.current_player = 0;
    next
}

fn advance_round_or_terminal(state: &LeducState) -> LeducState {
    let mut next = state.clone();
    if state.round_index == 0 {
        next.round_index = 1;
        next.current_player = 0;
        next.round_contributions = [0, 0];
        next.raises_in_round = 0;
    }
    next
}

fn terminal_utility(state: &LeducState) -> Option<f64> {
    if let Some(folded_player) = state.folded_player {
        let winner = 1 - folded_player;
        return Some(if winner == 0 {
            state.contributions[1] as f64
        } else {
            -(state.contributions[0] as f64)
        });
    }

    let public_card = state.public_card.as_ref()?;
    let round_history = &state.round_histories[1];
    if round_history.ends_with("xx") || round_history.ends_with("bc") || round_history.ends_with("rc") {
        return Some(showdown_utility(state, public_card));
    }
    None
}

fn showdown_utility(state: &LeducState, public_card: &str) -> f64 {
    let player_0_rank = card_rank(&state.private_cards[0]);
    let player_1_rank = card_rank(&state.private_cards[1]);
    let public_rank = card_rank(public_card);
    let player_0_pair = player_0_rank == public_rank;
    let player_1_pair = player_1_rank == public_rank;

    let winner: Option<usize> = if player_0_pair && !player_1_pair {
        Some(0)
    } else if player_1_pair && !player_0_pair {
        Some(1)
    } else if player_0_rank > player_1_rank {
        Some(0)
    } else if player_1_rank > player_0_rank {
        Some(1)
    } else {
        None
    };

    match winner {
        Some(0) => state.contributions[1] as f64,
        Some(1) => -(state.contributions[0] as f64),
        Some(_) => unreachable!(),
        None => (state.contributions[0] + state.contributions[1]) as f64 / 2.0 - state.contributions[0] as f64,
    }
}

fn state_infoset_key(state: &LeducState) -> String {
    let private_rank = state.private_cards[state.current_player].chars().next().unwrap();
    infoset_key(private_rank, state.public_card.as_ref().map(|card| card.chars().next().unwrap()), [&state.round_histories[0], &state.round_histories[1]])
}

fn infoset_key(private_rank: char, public_rank: Option<char>, round_histories: [&String; 2]) -> String {
    format!(
        "{}|{}|{}|{}",
        private_rank,
        public_rank.unwrap_or('-'),
        round_histories[0],
        round_histories[1]
    )
}

fn remaining_public_cards(state: &LeducState) -> Vec<&'static str> {
    DECK.iter()
        .copied()
        .filter(|card| card != &state.private_cards[0] && card != &state.private_cards[1])
        .collect()
}

fn private_deals() -> Vec<[&'static str; 2]> {
    let mut deals = Vec::new();
    for first in DECK {
        for second in DECK {
            if first != second {
                deals.push([first, second]);
            }
        }
    }
    deals
}

fn card_rank(card: &str) -> usize {
    match card.chars().next().unwrap() {
        'J' => 0,
        'Q' => 1,
        'K' => 2,
        _ => panic!("unexpected card"),
    }
}

pub fn train_leduc_cfr(iterations: usize) -> LeducTrainingSummary {
    let mut trainer = LeducTrainer::default();
    trainer.train(iterations)
}

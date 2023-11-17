use crate::Polarity::*;

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::ops::{Add, Index, IndexMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Polarity {
    Lambda,
    Xi,
    Epsilon,
    Phi,
    Zeta,
    Theta,
    Gamma,
    Omega,
}

const POLARITY_VARIANT_COUNT: usize = 8;

#[derive(Debug)]
struct Transformation {
    inputs: Vec<Polarity>,
    outputs: Vec<Polarity>,
}

impl Transformation {
    fn new(inputs: Vec<Polarity>, outputs: Vec<Polarity>) -> Self {
        Transformation { inputs, outputs }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ArcoState([u16; POLARITY_VARIANT_COUNT]);

impl ArcoState {
    fn new() -> Self {
        ArcoState([0; POLARITY_VARIANT_COUNT])
    }
}

impl Index<usize> for ArcoState {
    type Output = u16;
    fn index(&self, index: usize) -> &Self::Output {
        let ArcoState(self_internal) = self;
        &self_internal[index]
    }
}

impl IndexMut<usize> for ArcoState {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let ArcoState(self_internal) = self;
        &mut self_internal[index]
    }
}

impl Add for ArcoState {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let ArcoState(mut self_internal) = self;
        let ArcoState(rhs_internal) = rhs;
        for i in 0..self_internal.len() {
            self_internal[i] += rhs_internal[i];
        }
        ArcoState(self_internal)
    }
}

impl FromIterator<(Polarity, u16)> for ArcoState {
    fn from_iter<I: IntoIterator<Item = (Polarity, u16)>>(iter: I) -> Self {
        let ArcoState(mut result) = ArcoState::new();
        for (polarity, count) in iter {
            result[polarity as usize] += count;
        }
        ArcoState(result)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ArcoSearchState {
    state: ArcoState,
    catalysts: ArcoState,
}

impl From<ArcoState> for ArcoSearchState {
    fn from(value: ArcoState) -> Self {
        ArcoSearchState {
            state: value,
            catalysts: ArcoState::new(),
        }
    }
}

struct ScoredSearchState {
    search_state: ArcoSearchState,
    score: u16,
}

impl ScoredSearchState {
    fn new(search_state: ArcoSearchState, score: u16) -> Self {
        ScoredSearchState {
            search_state,
            score,
        }
    }
}

impl PartialEq for ScoredSearchState {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredSearchState {}

impl PartialOrd for ScoredSearchState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.score.cmp(&other.score))
    }
}

impl Ord for ScoredSearchState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

fn generate_transformed_state(
    initial_state: &ArcoSearchState,
    transformation: &Transformation,
) -> Option<ArcoSearchState> {
    let mut result = initial_state.clone();
    let mut initial_state_has_one_input = false;

    for polarity in &transformation.inputs {
        let i = *polarity as usize;
        if result.state[i] > 0 {
            initial_state_has_one_input = true;
            result.state[i] -= 1;
        } else {
            result.catalysts[i] += 1;
        }
    }

    for polarity in &transformation.outputs {
        let i = *polarity as usize;
        result.state[i] += 1;
    }

    if initial_state_has_one_input {
        Some(result)
    } else {
        None
    }
}

fn generate_neighbors(
    node: &ArcoSearchState,
    transformations: &[Transformation],
) -> Vec<ArcoSearchState> {
    let mut result = vec![];
    for t in transformations {
        if let Some(new_state) = generate_transformed_state(node, t) {
            result.push(new_state)
        }
    }
    result
}

fn path(
    ancestors: HashMap<ArcoSearchState, ArcoSearchState>,
    mut current_state: ArcoSearchState,
) -> Vec<ArcoSearchState> {
    let mut result = vec![current_state.clone()];
    while ancestors.contains_key(&current_state) {
        current_state = ancestors.get(&current_state).unwrap().clone();
        result.push(current_state.clone());
    }
    result
}

fn search(
    initial_state: ArcoState,
    goal_state: ArcoState,
    transformations: &[Transformation],
) -> Option<Vec<ArcoSearchState>> {
    let mut open_set: BinaryHeap<Reverse<ScoredSearchState>> = BinaryHeap::new();
    open_set.push(Reverse(ScoredSearchState::new(
        ArcoSearchState::from(initial_state.clone()),
        0,
    )));
    let mut ancestors: HashMap<ArcoSearchState, ArcoSearchState> = HashMap::new();
    let mut scores: HashMap<ArcoSearchState, u16> =
        HashMap::from([(ArcoSearchState::from(initial_state), 0)]);

    while !open_set.is_empty() {
        let current_state: ArcoSearchState = open_set.pop().unwrap().0.search_state;
        if current_state.state == goal_state.clone() + current_state.catalysts.clone() {
            return Some(path(ancestors, current_state));
        }
        for neighbor in generate_neighbors(&current_state, transformations) {
            let current_neighbor_score = scores.get(&current_state).unwrap() + 1;
            if !scores.contains_key(&neighbor)
                || current_neighbor_score < *scores.get(&neighbor).unwrap()
            {
                ancestors.insert(neighbor.clone(), current_state.clone());
                scores.insert(neighbor.clone(), current_neighbor_score);

                open_set.push(Reverse(ScoredSearchState::new(
                    neighbor,
                    current_neighbor_score,
                )));
                //TODO: avoid re-insertion
            }
        }
    }
    None
}

fn main() {
    let transformations = vec![
        Transformation::new(
            vec![Zeta, Theta, Gamma, Omega],
            vec![Lambda, Xi, Epsilon, Phi],
        ),
        Transformation::new(
            vec![Lambda, Xi, Epsilon, Phi],
            vec![Zeta, Theta, Gamma, Omega],
        ),
        Transformation::new(vec![Lambda, Omega], vec![Xi, Theta]),
        Transformation::new(vec![Xi, Gamma], vec![Zeta, Lambda]),
        Transformation::new(vec![Xi, Zeta], vec![Theta, Phi]),
        Transformation::new(vec![Lambda, Theta], vec![Epsilon, Zeta]),
        Transformation::new(vec![Theta, Epsilon], vec![Phi, Omega]),
        Transformation::new(vec![Zeta, Phi], vec![Gamma, Epsilon]),
        Transformation::new(vec![Phi, Gamma], vec![Omega, Xi]),
        Transformation::new(vec![Epsilon, Omega], vec![Lambda, Gamma]),
    ];

    println!("{:?}", transformations);

    let initial_state = ArcoState::from_iter([(Lambda, 2)]);
    let goal_state = ArcoState::from_iter([(Zeta, 1), (Omega, 1)]);

    println!("{:?}, {:?}", initial_state, goal_state);

    println!("{:?}", search(initial_state, goal_state, &transformations));
}

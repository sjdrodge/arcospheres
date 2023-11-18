use crate::Polarity::*;

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::ops::{Add, Index, IndexMut};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

const POLARITY_VARIANT_COUNT: usize = 8; // nightly: std::mem::variant_count<Polarity>()
const POLARITY_LIST: [Polarity; POLARITY_VARIANT_COUNT] =
    [Lambda, Xi, Epsilon, Phi, Zeta, Theta, Gamma, Omega];

impl fmt::Debug for Polarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Lambda => "λ",
                Xi => "ξ",
                Epsilon => "ε",
                Phi => "φ",
                Zeta => "ζ",
                Theta => "θ",
                Gamma => "γ",
                Omega => "ω",
            }
        )
    }
}

#[derive(Clone)]
struct Transformation {
    inputs: Vec<Polarity>,
    outputs: Vec<Polarity>,
}

impl fmt::Debug for Transformation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?} 🠖 {:?})", self.inputs, self.outputs)
    }
}

impl Transformation {
    fn new(inputs: Vec<Polarity>, outputs: Vec<Polarity>) -> Self {
        Transformation { inputs, outputs }
    }
}

type ArcoStateInt = u8;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ArcoState([ArcoStateInt; POLARITY_VARIANT_COUNT]);

// FIXME: holy moly this is disgusting, have some pride.
impl fmt::Display for ArcoState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut is_first_item = true;
        let ArcoState(internal_self) = self;
        write!(f, "{{")?;
        for i in 0..internal_self.len() {
            if internal_self[i] > 0 {
                if !is_first_item {
                    write!(f, ", ")?;
                }
                if internal_self[i] > 1 {
                    write!(f, "{}", internal_self[i])?;
                }
                write!(f, "{:?}", POLARITY_LIST[i])?;
                is_first_item = false;
            }
        }
        write!(f, "}}")?;
        Ok(())
    }
}

impl ArcoState {
    fn new() -> Self {
        ArcoState([0; POLARITY_VARIANT_COUNT])
    }
}

impl Index<usize> for ArcoState {
    type Output = ArcoStateInt;
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

impl FromIterator<(Polarity, ArcoStateInt)> for ArcoState {
    fn from_iter<I: IntoIterator<Item = (Polarity, ArcoStateInt)>>(iter: I) -> Self {
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

type ScoreInt = u8;
// TODO: become generic over score functions
struct ScoredSearchState {
    search_state: ArcoSearchState,
    score: ScoreInt,
}

impl ScoredSearchState {
    fn new(search_state: ArcoSearchState, score: ScoreInt) -> Self {
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

fn parity(state: &ArcoState) -> (ArcoStateInt, ArcoStateInt) {
    let parity1 = (state[0] % 4 + state[1] % 4 + state[2] % 4 + state[3] % 4) % 4;
    let parity2 = (state[4] % 4 + state[5] % 4 + state[6] % 4 + state[7] % 4) % 4;
    (parity1, parity2)
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
) -> Vec<(ArcoSearchState, Transformation)> {
    let mut result = vec![];
    for t in transformations {
        if let Some(new_state) = generate_transformed_state(node, t) {
            result.push((new_state, t.clone())) // TODO: this clone is ridiculous
        }
    }
    result
}

struct Path {
    initial_state: ArcoState,
    final_state: ArcoState,
    transformations: Vec<Transformation>,
}

fn path(
    ancestors: HashMap<ArcoSearchState, (ArcoSearchState, Transformation)>,
    mut current_state: ArcoSearchState,
) -> Path {
    let mut result = Vec::new();
    let final_state = current_state.clone();
    while ancestors.contains_key(&current_state) {
        let (s, t) = ancestors.get(&current_state).unwrap().clone();
        current_state = s;
        result.push(t);
    }
    result.reverse();
    Path {
        initial_state: current_state.state + final_state.catalysts,
        final_state: final_state.state,
        transformations: result,
    }
}

fn search(start: ArcoState, goal: ArcoState, transformations: &[Transformation]) -> Option<Path> {
    if parity(&start) != parity(&goal) {
        return None;
    }

    let mut open_set: BinaryHeap<Reverse<ScoredSearchState>> = BinaryHeap::new();
    open_set.push(Reverse(ScoredSearchState::new(
        ArcoSearchState::from(start.clone()),
        0,
    )));
    let mut ancestors: HashMap<ArcoSearchState, (ArcoSearchState, Transformation)> = HashMap::new();
    let mut scores: HashMap<ArcoSearchState, ScoreInt> =
        HashMap::from([(ArcoSearchState::from(start), 0)]);

    while !open_set.is_empty() {
        let current: ArcoSearchState = open_set.pop().unwrap().0.search_state;
        if current.state == goal.clone() + current.catalysts.clone() {
            return Some(path(ancestors, current));
        }
        for (neighbor, transformation) in generate_neighbors(&current, transformations) {
            let current_neighbor_score = scores.get(&current).unwrap() + 1;
            if !scores.contains_key(&neighbor)
                || current_neighbor_score < *scores.get(&neighbor).unwrap()
            {
                ancestors.insert(neighbor.clone(), (current.clone(), transformation));
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

fn print_search_result(path: Option<Path>) {
    if let Some(path) = path {
        println!("initial state: {}", path.initial_state,);
        println!("final state: {}", path.final_state);
        println!("path: {:#?}", path.transformations);
    } else {
        println!("No solution found.");
    }
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

    let start = ArcoState::from_iter([(Lambda, 4)]);
    let goal = ArcoState::from_iter([(Zeta, 2), (Omega, 2)]);
    println!("objective: {} 🠖 {}", start, goal);
    print_search_result(search(start, goal, &transformations));

    let start = ArcoState::from_iter([(Phi, 4)]);
    let goal = ArcoState::from_iter([(Zeta, 2), (Omega, 2)]);
    println!("objective: {} 🠖 {}", start, goal);
    print_search_result(search(start, goal, &transformations));

    let start = ArcoState::from_iter([(Lambda, 2)]);
    let goal = ArcoState::from_iter([(Phi, 2)]);
    println!("objective: {} 🠖 {}", start, goal);
    print_search_result(search(start, goal, &transformations));

    let start = ArcoState::from_iter([(Phi, 2)]);
    let goal = ArcoState::from_iter([(Lambda, 2)]);
    println!("objective: {} 🠖 {}", start, goal);
    print_search_result(search(start, goal, &transformations));
}

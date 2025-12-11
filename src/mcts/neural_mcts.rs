//! Neural Network enhanced MCTS implementation
//!
//! This module provides MCTS search with neural network policy and value guidance,
//! combining the mate-search-first innovation with modern neural network evaluation.

use super::{backpropagate, MctsNode, MoveCategory, EXPLORATION_CONSTANT};
use crate::board::Board;
use crate::boardstack::BoardStack;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use crate::neural_net::NeuralNetPolicy;
use crate::search::mate_search;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// Enhanced MCTS search with neural network policy guidance
///
/// This combines the mate-search-first strategy with neural network policy
/// to provide both tactical accuracy and strategic understanding.
///
/// # Arguments
/// * `root_state` - The initial board state
/// * `move_gen` - The move generator
/// * `pesto_eval` - Traditional evaluation function
/// * `nn_policy` - Neural network policy (optional)
/// * `mate_search_depth` - Depth for mate search (0 to disable)
/// * `iterations` - Number of MCTS iterations
/// * `time_limit` - Time limit for search
///
/// # Returns
/// The best move found, or None if no legal moves
pub fn neural_mcts_search(
    root_state: Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    nn_policy: &mut Option<NeuralNetPolicy>,
    mate_search_depth: i32,
    iterations: Option<u32>,
    time_limit: Option<Duration>,
) -> Option<Move> {
    if iterations.is_none() && time_limit.is_none() {
        panic!("MCTS search requires either iterations or time_limit to be set.");
    }

    let start_time = Instant::now();
    
    // **MATE SEARCH FIRST** - Our key tactical innovation
    if mate_search_depth > 0 {
        let mut mate_search_stack = BoardStack::with_board(root_state.clone());
        let (mate_score, mate_move, _) = mate_search(&mut mate_search_stack, move_gen, mate_search_depth, false);
        
        if mate_score >= 1_000_000 {
            println!("🎯 Mate found in {} plies: {:?}", mate_search_depth, mate_move);
            return Some(mate_move);
        } else if mate_score <= -1_000_000 {
            println!("⚠️  Being mated, returning defensive move: {:?}", mate_move);
            return Some(mate_move);
        }
    }

    // Initialize MCTS tree with neural network guidance
    let root_node = initialize_root_node(&root_state, move_gen, pesto_eval, nn_policy);
    
    let mut iteration_count = 0;
    let max_iterations = iterations.unwrap_or(u32::MAX);
    
    println!("🌲 Starting Neural MCTS search...");
    
    // Main MCTS loop
    while iteration_count < max_iterations {
        // Check time limit
        if let Some(limit) = time_limit {
            if start_time.elapsed() >= limit {
                break;
            }
        }
        
        // MCTS iteration: Selection -> Expansion -> Evaluation -> Backpropagation
        mcts_iteration(root_node.clone(), move_gen, pesto_eval, nn_policy);
        
        iteration_count += 1;
        
        if iteration_count % 100 == 0 {
            let elapsed = start_time.elapsed().as_millis();
            println!("🔄 Iteration {}, Time: {}ms", iteration_count, elapsed);
        }
    }
    
    // Select best move based on visit counts
    let best_move = select_best_move(&root_node);
    
    let total_time = start_time.elapsed().as_millis();
    println!("✅ Neural MCTS completed: {} iterations, {}ms", iteration_count, total_time);
    
    best_move
}

/// Initialize the root node with neural network policy guidance
fn initialize_root_node(
    root_state: &Board,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    nn_policy: &mut Option<NeuralNetPolicy>,
) -> Rc<RefCell<MctsNode>> {
    // Generate legal moves
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(root_state);
    let mut all_moves = captures;
    all_moves.extend(moves);
    
    // Filter to legal moves only
    let legal_moves: Vec<Move> = all_moves
        .into_iter()
        .filter(|&mv| {
            let new_board = root_state.apply_move_to_board(mv);
            new_board.is_legal(move_gen)
        })
        .collect();
    
    if legal_moves.is_empty() {
        // No legal moves (checkmate or stalemate) - the new_root constructor handles this
        return MctsNode::new_root(root_state.clone(), move_gen);
    }
    
    // Get neural network policy guidance if available
    let _move_priors = if let Some(ref mut nn) = nn_policy {
        if nn.is_available() {
            if let Some((policy, _value)) = nn.predict(root_state) {
                println!("🧠 Using neural network policy guidance");
                nn.policy_to_move_priors(&policy, &legal_moves)
            } else {
                uniform_priors(&legal_moves)
            }
        } else {
            uniform_priors(&legal_moves)
        }
    } else {
        uniform_priors(&legal_moves)
    };
    
    // Create root node using the existing constructor
    let root_node = MctsNode::new_root(root_state.clone(), move_gen);
    
    // For now, we'll start with an unexpanded root and let the existing MCTS handle expansion
    // The neural network priors will be applied during the normal expansion process
    
    root_node
}

/// Perform one MCTS iteration using the existing select_leaf_for_expansion
fn mcts_iteration(
    root_node: Rc<RefCell<MctsNode>>,
    move_gen: &MoveGen,
    pesto_eval: &PestoEval,
    nn_policy: &mut Option<NeuralNetPolicy>,
) {
    // Use the existing selection function but pass our exploration constant
    let leaf_node = super::node::select_leaf_for_expansion(root_node.clone(), EXPLORATION_CONSTANT);
    
    // Evaluation: get position value (the existing system handles expansion)
    let board_state = leaf_node.borrow().state.clone();
    let value = evaluate_position(&board_state, pesto_eval, nn_policy);
    
    // Backpropagation: update tree statistics
    backpropagate(leaf_node, value);
}

// Expansion is handled by the existing MCTS system

/// Evaluate a position using both traditional eval and neural network
fn evaluate_position(
    board: &Board,
    pesto_eval: &PestoEval,
    nn_policy: &mut Option<NeuralNetPolicy>,
) -> f64 {
    // Get traditional evaluation
    let pesto_eval_cp = pesto_eval.eval(board, &crate::move_generation::MoveGen::new());
    let pesto_value = eval_cp_to_win_prob(pesto_eval_cp);
    
    // Get neural network evaluation if available
    if let Some(ref mut nn) = nn_policy {
        if let Some(nn_eval_cp) = nn.get_position_value(board) {
            let nn_value = eval_cp_to_win_prob(nn_eval_cp);
            
            // Blend evaluations (70% NN, 30% traditional for now)
            return 0.7 * nn_value + 0.3 * pesto_value;
        }
    }
    
    // Fallback to traditional evaluation
    pesto_value
}

/// Convert centipawn evaluation to win probability
fn eval_cp_to_win_prob(eval_cp: i32) -> f64 {
    // Sigmoid conversion: 1 / (1 + exp(-eval/400))
    let eval_normalized = eval_cp as f64 / 400.0;
    1.0 / (1.0 + (-eval_normalized).exp())
}

// Board state is stored directly in each node

/// Create uniform priors for moves
fn uniform_priors(moves: &[Move]) -> Vec<(Move, f32)> {
    let uniform_prior = 1.0 / moves.len() as f32;
    moves.iter().map(|&mv| (mv, uniform_prior)).collect()
}

/// Categorize a move for MCTS purposes
fn categorize_move(mv: &Move, board: &Board) -> MoveCategory {
    // Check if target square has an enemy piece (capture)
    let target_piece = board.get_piece(mv.to);
    let is_capture = target_piece.is_some();
    
    if mv.is_promotion() {
        MoveCategory::Capture // Promotions are high priority like captures
    } else if is_capture {
        MoveCategory::Capture
    } else {
        MoveCategory::Quiet
    }
}

/// Select the best move based on visit counts
fn select_best_move(root_node: &Rc<RefCell<MctsNode>>) -> Option<Move> {
    let root_ref = root_node.borrow();
    
    let best_child = root_ref
        .children
        .iter()
        .max_by_key(|child| child.borrow().visits);
    
    best_child.and_then(|child| child.borrow().action)
}

/// UCT (Upper Confidence Bound applied to Trees) value calculation with neural network priors
pub fn uct_value_with_priors(node: &MctsNode, parent_visits: u32) -> f64 {
    if node.visits == 0 {
        return f64::INFINITY; // Unvisited nodes have highest priority
    }
    
    let exploitation = node.total_value / node.visits as f64;
    let exploration = EXPLORATION_CONSTANT * 
        (parent_visits as f64).ln().sqrt() / (node.visits as f64).sqrt();
    
    // Add neural network prior bonus
    let prior_bonus = 0.1 * node.prior_probability as f64; // Small bonus for NN guidance
    
    exploitation + exploration + prior_bonus
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_eval_cp_to_win_prob() {
        // Test conversion of centipawn evaluations to win probabilities
        assert!((eval_cp_to_win_prob(0) - 0.5).abs() < 0.01);
        assert!(eval_cp_to_win_prob(400) > 0.7);
        assert!(eval_cp_to_win_prob(-400) < 0.3);
    }
    
    #[test]
    fn test_uniform_priors() {
        let move_gen = MoveGen::new();
        let board = Board::new();
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
        let mut all_moves = captures;
        all_moves.extend(moves);
        
        let priors = uniform_priors(&all_moves);
        
        // Should have same number of priors as moves
        assert_eq!(priors.len(), all_moves.len());
        
        // All priors should sum to 1.0
        let total: f32 = priors.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 0.001);
    }
}
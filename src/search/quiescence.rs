use super::see::see;
use crate::boardstack::BoardStack;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use std::time::{Duration, Instant};

/// Tactical result from Quiescence Search for MCTS grafting
#[derive(Clone, Debug)]
pub struct TacticalTree {
    pub principal_variation: Vec<Move>,
    pub leaf_score: i32,
    pub siblings: Vec<(Move, i32)>, // Other tactical moves at root and their scores
}

/// Performs a quiescence search to evaluate tactical sequences and avoid the horizon effect.
pub fn quiescence_search(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    mut alpha: i32,
    beta: i32,
    max_depth: i32, // Remaining q-search depth
    _verbose: bool,
    start_time: Option<Instant>,
    time_limit: Option<Duration>,
) -> (i32, i32) {
    let mut nodes = 1;

    if let (Some(start), Some(limit)) = (start_time, time_limit) {
        if start.elapsed() >= limit {
             let stand_pat = pesto.eval(&board.current_state(), move_gen);
             return (stand_pat, nodes);
        }
    }

    let stand_pat = pesto.eval(&board.current_state(), move_gen);

    if stand_pat >= beta {
        return (beta, nodes);
    }

    if stand_pat > alpha {
        alpha = stand_pat;
    }

    if max_depth <= 0 {
        return (alpha, nodes);
    }

    let captures = move_gen.gen_pseudo_legal_captures(&board.current_state());
    if captures.is_empty() && !board.is_check(move_gen) {
        return (alpha, nodes);
    }

    for capture in captures {
        if see(&board.current_state(), move_gen, capture.to, capture.from) < 0 {
            continue;
        }

        board.make_move(capture);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }

        let (mut score, n) = quiescence_search(
            board,
            move_gen,
            pesto,
            -beta,
            -alpha,
            max_depth - 1,
            _verbose,
            start_time,
            time_limit,
        );
        score = -score;
        nodes += n;

        board.undo_move();

        if score >= beta {
            return (beta, nodes);
        }
        if score > alpha {
            alpha = score;
        }
    }

    (alpha, nodes)
}

/// Specialized Quiescence Search for Tier 2 MCTS Integration
/// Returns the full TacticalTree instead of just a score.
pub fn quiescence_search_tactical(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
) -> TacticalTree {
    let mut siblings = Vec::new();
    let stand_pat = pesto.eval(&board.current_state(), move_gen);
    let mut best_score = stand_pat;
    let mut best_pv = Vec::new();

    let captures = move_gen.gen_pseudo_legal_captures(&board.current_state());
    
    for capture in captures {
        board.make_move(capture);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }

        // Full search for the first level to find best tactical line
        let (score, _nodes) = quiescence_search(
            board,
            move_gen,
            pesto,
            -1000001,
            1000001,
            8,
            false,
            None,
            None,
        );
        let score = -score;
        board.undo_move();

        siblings.push((capture, score));

        if score > best_score {
            best_score = score;
            best_pv = vec![capture];
        }
    }

    TacticalTree {
        principal_variation: best_pv,
        leaf_score: best_score,
        siblings,
    }
}
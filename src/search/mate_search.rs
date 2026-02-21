//! Mate search algorithm with configurable exhaustive depth.
//!
//! Searches for forced mates using iterative deepening with pure minimax.
//! The `exhaustive_depth` parameter controls which depths use exhaustive search
//! (all legal moves) vs checks-only search (only checking moves):
//!
//! - depth ≤ exhaustive_depth: **exhaustive** — all legal attacker moves tried
//! - depth > exhaustive_depth: **checks-only** — only checking moves on attacker plies
//!
//! With the default exhaustive_depth=3, this gives:
//! - Mate-in-1 (depth 1): exhaustive
//! - Mate-in-2 (depth 3): exhaustive — catches quiet-first mates like 1.Qg7! Kh8 2.Qh7#
//! - Mate-in-3 (depth 5): checks-only — keeps branching manageable
//!
//! On the defender's plies all legal moves are always tried.
//!
//! # Integration with MCTS
//!
//! This module provides the **Tier 1 Safety Gate** in the three-tier MCTS
//! architecture. Before expanding any MCTS node, the engine checks for forced
//! mates:
//!
//! - If a forced win is found, the move is played immediately (no MCTS needed)
//! - Results are cached in the transposition table to avoid redundant searches
//!
//! # Score Convention
//!
//! - `1_000_000 + depth`: Forced mate in `depth` plies (winning)
//! - `0`: No forced mate found, or draw
//!
//! # Example
//!
//! ```ignore
//! use kingfisher::search::mate_search;
//! use kingfisher::board::Board;
//! use kingfisher::move_generation::MoveGen;
//!
//! let board = Board::new();
//! let move_gen = MoveGen::new();
//!
//! // Search for mate up to depth 6 (3 moves each side)
//! let (score, best_move, nodes) = mate_search(&board, &move_gen, 6, false, 3);
//!
//! if score >= 1_000_000 {
//!     println!("Forced mate found! Play: {:?}", best_move);
//! }
//! ```

use crate::board::Board;
use crate::move_generation::MoveGen;
use crate::move_types::Move;

/// Public API: Mate search with configurable exhaustive depth.
///
/// Searches for forced mates using iterative deepening. Depths ≤ `exhaustive_depth`
/// use exhaustive search (all legal attacker moves), while deeper levels use
/// checks-only search. Default exhaustive_depth=3 makes mate-in-1 and mate-in-2
/// exhaustive, and mate-in-3 checks-only.
///
/// Stateless: operates on immutable `&Board` references (like KOTH search),
/// avoiding BoardStack overhead. Repetition detection is unnecessary — forced
/// mates exist regardless of position history.
pub fn mate_search(
    board: &Board,
    move_gen: &MoveGen,
    max_depth: i32,
    _verbose: bool,
    exhaustive_depth: i32,
) -> (i32, Move, i32) {
    let mut nodes: i32 = 0;

    for d in 1..=max_depth {
        let depth = 2 * d - 1; // Only check odd depths (mate for us)
        let checks_only = depth > exhaustive_depth;

        let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);

        for m in captures.iter().chain(moves.iter()) {
            // On attacker turn with checks_only: filter before legality/clone
            if checks_only && !board.gives_check(*m, move_gen) {
                continue;
            }

            if !board.is_legal_after_move(*m, move_gen) {
                continue;
            }

            let next_board = board.apply_move_to_board(*m);

            if solve_mate(&next_board, move_gen, depth - 1, false, checks_only, &mut nodes) {
                return (1_000_000 + depth, *m, nodes);
            }
        }
    }

    (0, Move::null(), nodes)
}

/// Recursive pure minimax mate solver.
///
/// Returns true if the position is a forced mate for the attacker.
/// - Attacker: returns true if ANY child leads to mate (short-circuits on first success)
/// - Defender: returns true only if ALL children lead to mate (short-circuits on first refutation)
fn solve_mate(
    board: &Board,
    move_gen: &MoveGen,
    depth: i32,
    is_attackers_turn: bool,
    checks_only: bool,
    nodes: &mut i32,
) -> bool {
    *nodes += 1;

    // Depth 0: check for checkmate
    if depth <= 0 {
        if !board.is_check(move_gen) {
            return false; // Not in check = not checkmate
        }
        // In check — verify no legal escape exists
        let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
        let has_legal = captures
            .iter()
            .chain(moves.iter())
            .any(|m| board.is_legal_after_move(*m, move_gen));
        return !has_legal; // Checkmate if no legal moves
    }

    let (captures, moves) = move_gen.gen_pseudo_legal_moves(board);
    let mut has_legal_move = false;

    for m in captures.iter().chain(moves.iter()) {
        // On attacker turns with checks_only: filter before legality/clone
        if is_attackers_turn && checks_only && !board.gives_check(*m, move_gen) {
            continue;
        }

        if !board.is_legal_after_move(*m, move_gen) {
            continue;
        }

        has_legal_move = true;
        let next_board = board.apply_move_to_board(*m);

        let child_result = solve_mate(
            &next_board,
            move_gen,
            depth - 1,
            !is_attackers_turn,
            checks_only,
            nodes,
        );

        if is_attackers_turn {
            if child_result {
                return true; // Found a mating line
            }
        } else {
            // Defender
            if !child_result {
                return false; // Found a refutation
            }
        }
    }

    // No legal moves found
    if !has_legal_move {
        if is_attackers_turn && checks_only {
            // No checking moves available — non-checking moves may exist
            return false;
        }
        // Truly no legal moves: checkmate or stalemate
        return board.is_check(move_gen); // true = checkmate, false = stalemate
    }

    // Attacker: no child succeeded → no mate
    // Defender: all children led to mate → mate is forced
    !is_attackers_turn
}

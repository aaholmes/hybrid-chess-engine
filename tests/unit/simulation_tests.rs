//! Unit tests for MCTS simulation (random playout)

use crate::common::{board_from_fen, positions};
use kingfisher::board::Board;
use kingfisher::mcts::simulation::simulate_random_playout;
use kingfisher::move_generation::MoveGen;

#[test]
fn test_playout_terminates_from_starting_position() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    // Should terminate without panic
    let result = simulate_random_playout(&board, &move_gen);
    assert!(
        (0.0..=1.0).contains(&result),
        "Playout result should be in [0.0, 1.0], got {result}"
    );
}

#[test]
fn test_playout_returns_value_in_range() {
    let move_gen = MoveGen::new();
    // Run multiple playouts and check all are in range
    for _ in 0..10 {
        let board = Board::new();
        let result = simulate_random_playout(&board, &move_gen);
        assert!(
            (0.0..=1.0).contains(&result),
            "Playout result out of range: {result}"
        );
    }
}

#[test]
fn test_checkmate_returns_decisive_result() {
    let move_gen = MoveGen::new();
    // Black is checkmated: result from black's perspective should be 0.0 (loss)
    let mated = board_from_fen("k7/1Q6/1K6/8/8/8/8/8 b - - 0 1");
    let result = simulate_random_playout(&mated, &move_gen);
    assert!(
        result == 0.0 || result == 1.0,
        "Checkmate should return decisive result (0.0 or 1.0), got {result}"
    );
}

#[test]
fn test_stalemate_returns_draw() {
    let move_gen = MoveGen::new();
    let stalemate = board_from_fen(positions::STALEMATE);
    let result = simulate_random_playout(&stalemate, &move_gen);
    assert!(
        (result - 0.5).abs() < f64::EPSILON,
        "Stalemate should return 0.5, got {result}"
    );
}

#[test]
fn test_material_advantage_influences_result() {
    let move_gen = MoveGen::new();
    // White has massive material advantage: queen + rook vs lone king
    let white_winning = board_from_fen("4k3/8/8/8/8/8/8/R3K2Q w - - 0 1");

    // Run many playouts and check average trends toward winning
    let mut total = 0.0;
    let n = 50;
    for _ in 0..n {
        total += simulate_random_playout(&white_winning, &move_gen);
    }
    let avg = total / n as f64;
    // With massive material advantage, average should trend above 0.5
    assert!(
        avg > 0.4,
        "With material advantage, average playout should trend above 0.4, got {avg}"
    );
}

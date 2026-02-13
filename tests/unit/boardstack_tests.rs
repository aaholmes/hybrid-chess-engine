//! Unit tests for BoardStack (move history and undo/redo)

use crate::common::positions;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_types::Move;
use kingfisher::piece_types::{PAWN, WHITE};

#[test]
fn test_new_starts_at_initial_position() {
    let stack = BoardStack::new();
    let board = stack.current_state();
    assert!(board.w_to_move, "Starting position should be white to move");
    // position_history should have exactly 1 entry
    assert_eq!(
        stack.position_history.len(),
        1,
        "Should have one position in history"
    );
}

#[test]
fn test_new_from_fen() {
    let stack = BoardStack::new_from_fen(positions::EN_PASSANT);
    let board = stack.current_state();
    assert_eq!(
        board.en_passant(),
        Some(40),
        "En passant square should be a6 (40)"
    );
}

#[test]
fn test_make_move_and_undo_restores_state() {
    let mut stack = BoardStack::new();
    let original_fen = stack.current_state().to_fen().unwrap();

    // e2e4
    let mv = Move::new(12, 28, None);
    stack.make_move(mv);
    let after_fen = stack.current_state().to_fen().unwrap();
    assert_ne!(after_fen, original_fen, "FEN should change after move");
    assert!(
        !stack.current_state().w_to_move,
        "Should be black to move after e2e4"
    );

    // Undo
    let undone = stack.undo_move();
    assert_eq!(undone, Some(mv));
    assert_eq!(
        stack.current_state().to_fen().unwrap(),
        original_fen,
        "FEN should be restored after undo"
    );
    assert!(
        stack.current_state().w_to_move,
        "Should be white to move after undo"
    );
}

#[test]
fn test_current_state_returns_latest() {
    let mut stack = BoardStack::new();
    // Make move e2e4
    stack.make_move(Move::new(12, 28, None));
    let state = stack.current_state();
    // White pawn should now be on e4, not e2
    assert_eq!(
        state.get_piece(28),
        Some((WHITE, PAWN)),
        "Pawn should be on e4"
    );
    assert_eq!(state.get_piece(12), None, "e2 should be empty");
}

#[test]
fn test_is_draw_by_repetition_true() {
    let mut stack = BoardStack::new();

    // Play Nf3-Ng1 / Nf6-Ng8 twice to reach starting position 3 times
    let nf3 = Move::new(6, 21, None); // g1-f3
    let nf6 = Move::new(62, 45, None); // g8-f6
    let ng1 = Move::new(21, 6, None); // f3-g1
    let ng8 = Move::new(45, 62, None); // f6-g8

    // Cycle 1
    stack.make_move(nf3);
    stack.make_move(nf6);
    stack.make_move(ng1);
    stack.make_move(ng8);
    // Back to starting position (2nd time)
    assert!(
        !stack.is_draw_by_repetition(),
        "Should not be draw after 2 repetitions"
    );

    // Cycle 2
    stack.make_move(nf3);
    stack.make_move(nf6);
    stack.make_move(ng1);
    stack.make_move(ng8);
    // Back to starting position (3rd time)
    assert!(
        stack.is_draw_by_repetition(),
        "Should be draw after 3 repetitions"
    );
}

#[test]
fn test_is_draw_by_repetition_false_different_moves() {
    let mut stack = BoardStack::new();

    // Play different moves that don't repeat position
    stack.make_move(Move::new(12, 28, None)); // e2e4
    stack.make_move(Move::new(52, 36, None)); // e7e5
    assert!(!stack.is_draw_by_repetition());
}

#[test]
fn test_multiple_make_undo_cycles() {
    let mut stack = BoardStack::new();
    let original_fen = stack.current_state().to_fen().unwrap();

    // Make and undo several moves
    for _ in 0..5 {
        stack.make_move(Move::new(12, 28, None)); // e2e4
        stack.undo_move();
        assert_eq!(stack.current_state().to_fen().unwrap(), original_fen);
    }
}

#[test]
fn test_null_move_make_and_undo() {
    let mut stack = BoardStack::new();
    let original_fen = stack.current_state().to_fen().unwrap();
    assert!(stack.current_state().w_to_move);

    stack.make_null_move();
    assert!(
        !stack.current_state().w_to_move,
        "Null move should flip side to move"
    );
    assert_eq!(
        stack.current_state().en_passant(),
        None,
        "Null move should clear en passant"
    );

    stack.undo_null_move();
    assert!(
        stack.current_state().w_to_move,
        "Undo null move should restore side to move"
    );
    assert_eq!(
        stack.current_state().to_fen().unwrap(),
        original_fen,
        "FEN should be restored after undo null move"
    );
}

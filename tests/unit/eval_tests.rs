//! Unit tests for the Pesto evaluation function

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::piece_types::{KNIGHT, PAWN, WHITE};
use crate::common::{board_from_fen, positions};

#[test]
fn test_pesto_eval_new_initializes() {
    let eval = PestoEval::new();
    // Verify tables are populated: white pawn on e2 should have nonzero mg score
    let mg = eval.get_mg_score(WHITE, PAWN, 12); // e2
    assert_ne!(mg, 0, "Middlegame score for white pawn on e2 should be nonzero");
}

#[test]
fn test_starting_position_eval_near_zero() {
    let eval = PestoEval::new();
    let board = Board::new();
    let move_gen = MoveGen::new();
    let score = eval.eval(&board, &move_gen);
    // Starting position should be roughly equal (within a small margin)
    assert!(
        score.abs() < 100,
        "Starting position eval should be near zero, got {score}"
    );
}

#[test]
fn test_material_advantage_positive() {
    let eval = PestoEval::new();
    let move_gen = MoveGen::new();

    // White up a queen
    let board = board_from_fen(positions::WHITE_UP_QUEEN);
    let score = eval.eval(&board, &move_gen);
    assert!(score > 0, "White up a queen should have positive eval, got {score}");

    // Black up a queen (white to move, so eval is from white's perspective)
    let board_b = board_from_fen(positions::BLACK_UP_QUEEN);
    let score_b = eval.eval(&board_b, &move_gen);
    assert!(score_b < 0, "Black up a queen should have negative eval for white, got {score_b}");
}

#[test]
fn test_tapered_eval_game_phase() {
    let eval = PestoEval::new();
    let move_gen = MoveGen::new();

    // Starting position has high game phase (many pieces)
    let board_start = Board::new();
    let (_, _, phase_start) = eval.eval_plus_game_phase(&board_start, &move_gen);
    assert!(phase_start > 20, "Starting position should have high game phase, got {phase_start}");

    // Kings-only endgame has zero game phase
    let board_end = board_from_fen(positions::EQUAL_MATERIAL);
    let (_, _, phase_end) = eval.eval_plus_game_phase(&board_end, &move_gen);
    assert_eq!(phase_end, 0, "Kings-only position should have game phase 0");
}

#[test]
fn test_knight_center_vs_rim() {
    let eval = PestoEval::new();
    // Knight on e4 (28) should have better piece-square score than on a1 (0)
    let center_mg = eval.get_mg_score(WHITE, KNIGHT, 28);
    let rim_mg = eval.get_mg_score(WHITE, KNIGHT, 0);
    assert!(
        center_mg > rim_mg,
        "Knight on e4 ({center_mg}) should score higher than on a1 ({rim_mg})"
    );
}

#[test]
fn test_passed_pawn_bonus_increases_eval() {
    let eval = PestoEval::new();
    let move_gen = MoveGen::new();

    // White has a passed pawn on e6 with no black pawns blocking
    let passed = board_from_fen("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1");
    let no_pawn = board_from_fen(positions::EQUAL_MATERIAL);

    let score_passed = eval.eval(&passed, &move_gen);
    let score_no_pawn = eval.eval(&no_pawn, &move_gen);
    assert!(
        score_passed > score_no_pawn,
        "Position with passed pawn ({score_passed}) should eval higher than without ({score_no_pawn})"
    );
}

#[test]
fn test_king_safety_pawn_shield() {
    let eval = PestoEval::new();
    let move_gen = MoveGen::new();

    // White king with pawn shield vs without
    let with_shield = board_from_fen("4k3/8/8/8/8/8/5PPP/6K1 w - - 0 1");
    let without_shield = board_from_fen("4k3/8/8/8/8/8/8/6K1 w - - 0 1");

    let (mg_with, _, _) = eval.eval_plus_game_phase(&with_shield, &move_gen);
    let (mg_without, _, _) = eval.eval_plus_game_phase(&without_shield, &move_gen);
    // With pawn shield should have better middlegame evaluation
    assert!(
        mg_with > mg_without,
        "King with pawn shield mg ({mg_with}) should be better than without ({mg_without})"
    );
}

#[test]
fn test_symmetric_eval() {
    let eval = PestoEval::new();
    let move_gen = MoveGen::new();

    // Mirror position: white to move vs black to move with same setup
    let white_to_move = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let black_to_move = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");

    let score_w = eval.eval(&white_to_move, &move_gen);
    let score_b = eval.eval(&black_to_move, &move_gen);

    // Eval from the side to move's perspective should have same magnitude but sign may differ
    // due to tempo. They should at least be close.
    assert!(
        (score_w + score_b).abs() < 50,
        "Symmetric positions should have roughly opposite evals: w={score_w}, b={score_b}"
    );
}


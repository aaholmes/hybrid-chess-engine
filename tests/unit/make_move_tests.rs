//! Unit tests for apply_move_to_board (making moves on the board)

use kingfisher::board::Board;
use kingfisher::board_utils::sq_ind_to_bit;
use kingfisher::move_types::Move;
use kingfisher::piece_types::{KING, PAWN, QUEEN, ROOK, WHITE};
use crate::common::{board_from_fen, positions};

#[test]
fn test_standard_pawn_push() {
    let board = Board::new();
    // e2e3 (single push)
    let mv = Move::new(12, 20, None);
    let new_board = board.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(20), Some((WHITE, PAWN)), "Pawn should be on e3");
    assert_eq!(new_board.get_piece(12), None, "e2 should be empty");
    assert!(!new_board.w_to_move, "Should be black to move");
    assert_eq!(new_board.en_passant(), None, "Single push should not set en passant");
}

#[test]
fn test_double_pawn_push_sets_en_passant() {
    let board = Board::new();
    // e2e4 (double push)
    let mv = Move::new(12, 28, None);
    let new_board = board.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(28), Some((WHITE, PAWN)), "Pawn should be on e4");
    assert_eq!(new_board.get_piece(12), None, "e2 should be empty");
    assert_eq!(new_board.en_passant(), Some(20), "En passant should be set to e3 (20)");
}

#[test]
fn test_en_passant_capture() {
    // Position with white pawn on b5 and black pawn on a5 after double push, en passant on a6
    let board = board_from_fen(positions::EN_PASSANT);
    // b5xa6 en passant
    let mv = Move::new(33, 40, None); // b5 -> a6

    let new_board = board.apply_move_to_board(mv);
    assert_eq!(new_board.get_piece(40), Some((WHITE, PAWN)), "White pawn should be on a6");
    assert_eq!(new_board.get_piece(33), None, "b5 should be empty");
    // The captured pawn on a5 (32) should be removed
    assert_eq!(new_board.get_piece(32), None, "Captured pawn on a5 should be removed");
}

#[test]
fn test_kingside_castling() {
    let board = board_from_fen(positions::CASTLING_BOTH);
    // White kingside castle: e1g1
    let mv = Move::new(4, 6, None);
    let new_board = board.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(6), Some((WHITE, KING)), "King should be on g1");
    assert_eq!(new_board.get_piece(5), Some((WHITE, ROOK)), "Rook should be on f1");
    assert_eq!(new_board.get_piece(4), None, "e1 should be empty");
    assert_eq!(new_board.get_piece(7), None, "h1 should be empty");
    assert!(!new_board.castling_rights.white_kingside);
    assert!(!new_board.castling_rights.white_queenside);
}

#[test]
fn test_queenside_castling() {
    let board = board_from_fen(positions::CASTLING_BOTH);
    // White queenside castle: e1c1
    let mv = Move::new(4, 2, None);
    let new_board = board.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(2), Some((WHITE, KING)), "King should be on c1");
    assert_eq!(new_board.get_piece(3), Some((WHITE, ROOK)), "Rook should be on d1");
    assert_eq!(new_board.get_piece(4), None, "e1 should be empty");
    assert_eq!(new_board.get_piece(0), None, "a1 should be empty");
    assert!(!new_board.castling_rights.white_kingside);
    assert!(!new_board.castling_rights.white_queenside);
}

#[test]
fn test_pawn_promotion() {
    let board = board_from_fen(positions::PROMOTION);
    // a7a8=Q
    let mv = Move::new(48, 56, Some(QUEEN));
    let new_board = board.apply_move_to_board(mv);

    assert_eq!(new_board.get_piece(56), Some((WHITE, QUEEN)), "Should be a queen on a8");
    assert_eq!(new_board.get_piece(48), None, "a7 should be empty");
    // Verify the pawn bitboard no longer has the pawn
    assert_eq!(
        new_board.get_piece_bitboard(WHITE, PAWN) & sq_ind_to_bit(56),
        0,
        "No pawn should be on a8"
    );
    assert_ne!(
        new_board.get_piece_bitboard(WHITE, QUEEN) & sq_ind_to_bit(56),
        0,
        "Queen should be on a8 in queen bitboard"
    );
}

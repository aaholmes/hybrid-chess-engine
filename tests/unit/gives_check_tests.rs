/// Tests for gives_check: determines if a move gives check without full board clone.

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::piece_types::QUEEN;

use crate::common::{board_from_fen, generate_legal_moves};

/// Moving a queen to attack the king square gives check.
#[test]
fn test_gives_check_queen_to_e7() {
    // White queen on d1, black king on e8. Qd1-e2 doesn't check, but Qd1-d8 or similar might.
    // Use a clearer position: White queen d1, black king e8, clear d-file
    let board = board_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    let move_gen = MoveGen::new();

    // Qd1-d8 should give check (queen on d8 attacks e8)
    let qd8 = Move::new(3, 59, None); // d1 -> d8
    assert!(board.gives_check(qd8, &move_gen), "Qd8 should give check");

    // Qd1-a4 gives check via the a4-e8 diagonal
    let qa4 = Move::new(3, 24, None); // d1 -> a4
    assert!(board.gives_check(qa4, &move_gen), "Qa4 should give check via diagonal");

    // Qd1-a1 should NOT give check (a1 doesn't attack e8)
    let qa1 = Move::new(3, 0, None); // d1 -> a1
    assert!(!board.gives_check(qa1, &move_gen), "Qa1 should not give check");
}

/// Knight move that attacks king gives check.
#[test]
fn test_gives_check_knight_fork() {
    // White knight on d5, black king on e7
    let board = board_from_fen("8/4k3/8/3N4/8/8/8/4K3 w - - 0 1");
    let move_gen = MoveGen::new();

    // Nd5-f6 attacks e8... but king is on e7. Nd5-c7 attacks e8 and e6.
    // Actually let me use: knight on e5, king on g6. Ne5-f7 gives check? No.
    // Let me use: Nc3, king on e4. Nc3-d5 attacks e7, not e4.
    // Simplest: knight on c5, king on d7. Nc5-e6 doesn't check d7.
    // OK: knight on d2, king on e4 — doesn't help.

    // Simple: knight on g5, king on f7. Ng5-e6 checks f7? No, knight on e6 attacks d8,f8,d4,f4,c5,c7,g5,g7.
    // Ng5-h3 doesn't check.
    // Let me just use a position where it's obvious.
    // White Nf3, black Ke8. Nf3-g5 doesn't check e8.
    // White Nf5, black Ke7. Nf5-d6 checks? knight on d6 attacks b5,b7,c4,c8,e4,e8,f5,f7. Yes, attacks e8? No, king on e7.
    // Actually d6 attacks e8 and f7 and f5. King on e7 is not attacked.

    // Let me just set up Nc6 attacking Ke7:
    // Knight on c6 attacks: a5,a7,b4,b8,d4,d8,e5,e7. Yes! Attacks e7.
    let board2 = board_from_fen("8/4k3/8/8/8/2N5/8/4K3 w - - 0 1");
    // Nc3-b5 attacks a7,c7,a3,c3,d4,d6 — doesn't check e7
    // Nc3-d5 attacks b4,b6,c3,c7,e3,e7,f4,f6 — attacks e7! Check!
    let nd5 = Move::new(18, 35, None); // c3(18) -> d5(35)
    assert!(board2.gives_check(nd5, &move_gen), "Nd5 should give check to Ke7");

    // Nc3-a4 should NOT give check
    let na4 = Move::new(18, 24, None); // c3 -> a4
    assert!(!board2.gives_check(na4, &move_gen), "Na4 should not give check");
}

/// Discovered check: moving a piece reveals check from behind.
#[test]
fn test_gives_check_discovered_check() {
    // White rook on e1, white bishop on e4, black king on e8
    // Moving the bishop off the e-file reveals the rook's attack on e8
    let board = board_from_fen("4k3/8/8/8/4B3/8/8/4RK2 w - - 0 1");
    let move_gen = MoveGen::new();

    // Be4-d5 (moves off e-file, discovers check from Re1)
    let bd5 = Move::new(28, 35, None); // e4(28) -> d5(35)
    assert!(board.gives_check(bd5, &move_gen), "Bd5 should give discovered check");
}

/// Non-checking move returns false.
#[test]
fn test_gives_check_false_for_quiet_move() {
    let board = Board::new();
    let move_gen = MoveGen::new();

    // e2-e4 from starting position — definitely not check
    let e2e4 = Move::new(12, 28, None);
    assert!(!board.gives_check(e2e4, &move_gen), "e4 should not give check in starting position");

    // Nc3 from starting position
    let nc3 = Move::new(1, 18, None);
    assert!(!board.gives_check(nc3, &move_gen), "Nc3 should not give check in starting position");
}

/// Pawn advance that gives check.
#[test]
fn test_gives_check_pawn_check() {
    // White pawn on d6, black king on e7. d6-d7 gives check? Pawn attacks diagonals.
    // Pawn on d6 attacks c7 and e7. So the pawn on d6 already attacks e7.
    // But after d6-d7, pawn on d7 attacks c8 and e8. King on e7 is NOT attacked.
    // Better: pawn on f6, king on e7. f6 attacks e7 and g7. But we want a pawn MOVE to give check.
    // Pawn on f5 -> f6: pawn on f6 attacks e7 and g7. If king on e7, this is check!
    let board = board_from_fen("8/4k3/8/5P2/8/8/8/4K3 w - - 0 1");
    let move_gen = MoveGen::new();

    let f6 = Move::new(37, 45, None); // f5(37) -> f6(45)
    assert!(board.gives_check(f6, &move_gen), "f6 should give check to Ke7");
}

/// Property test: for all legal moves in several positions, gives_check matches
/// the result of apply_move + is_check.
#[test]
fn test_gives_check_matches_apply_move() {
    let positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // Starting
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",       // Castling available
        "8/P7/8/8/8/8/8/K6k w - - 0 1",                               // Promotion
        "4k3/8/8/3q4/4N3/8/8/K7 w - - 0 1",                           // Tactical
        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", // Italian
        "8/8/8/pP6/8/8/8/K6k w - a6 0 1",                             // En passant
        "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2", // After 1...Nc6
    ];

    let move_gen = MoveGen::new();

    for fen in &positions {
        let board = board_from_fen(fen);
        let legal_moves = generate_legal_moves(&board, &move_gen);

        for mv in &legal_moves {
            let gives_check_fast = board.gives_check(*mv, &move_gen);
            let new_board = board.apply_move_to_board(*mv);
            let gives_check_slow = new_board.is_check(&move_gen);

            assert_eq!(gives_check_fast, gives_check_slow,
                "gives_check mismatch for {} in position {}: fast={}, slow={}",
                mv.to_uci(), fen, gives_check_fast, gives_check_slow);
        }
    }
}

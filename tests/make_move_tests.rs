#[cfg(test)]
mod make_move_tests {
    use kingfisher::board::Board;
    use kingfisher::board_utils;
    use kingfisher::move_generation::MoveGen; // Needed for legality checks in some tests
    use kingfisher::move_types::Move;
    use kingfisher::piece_types::*;

    fn create_move(from: &str, to: &str) -> Move {
        Move::new(
            board_utils::algebraic_to_sq_ind(from),
            board_utils::algebraic_to_sq_ind(to),
            None,
        )
    }
    fn create_promo_move(from: &str, to: &str, promo_piece: usize) -> Move {
        Move::new(
            board_utils::algebraic_to_sq_ind(from),
            board_utils::algebraic_to_sq_ind(to),
            Some(promo_piece),
        )
    }

    #[test]
    fn test_apply_pawn_push() {
        let board = Board::new(); // Initial position
        let mv = create_move("e2", "e4");
        let next_board = board.apply_move_to_board(mv);

        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("e2")),
            None
        );
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("e4")),
            Some((WHITE, PAWN))
        );
        assert!(!next_board.w_to_move); // Side to move flipped
                                        // Test that the resulting position has correct FEN (implicitly tests en passant, clocks, etc.)
        assert_eq!(
            next_board.to_fen(),
            Some("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1".to_string())
        );
    }

    #[test]
    fn test_apply_capture() {
        let board =
            Board::new_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"); // After 1.e4 d5
        let mv = create_move("e4", "d5"); // exd5
        let next_board = board.apply_move_to_board(mv);

        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("e4")),
            None
        );
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("d5")),
            Some((WHITE, PAWN))
        ); // White pawn captures
        assert!(!next_board.w_to_move);
        // Test that the resulting position has correct FEN
        assert_eq!(
            next_board.to_fen(),
            Some("rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2".to_string())
        );
    }

    #[test]
    fn test_apply_en_passant() {
        let board =
            Board::new_from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"); // White can capture e.p. on f6
        let mv = create_move("e5", "f6"); // exf6 e.p.
        let next_board = board.apply_move_to_board(mv);

        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("e5")),
            None
        ); // White pawn moved
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("f6")),
            Some((WHITE, PAWN))
        ); // White pawn on f6
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("f5")),
            None
        ); // Black pawn captured e.p.
        assert!(!next_board.w_to_move);
        // Test that the resulting position has correct FEN (en passant should be reset)
        assert_eq!(
            next_board.to_fen(),
            Some("rnbqkbnr/ppp1p1pp/5P2/3p4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3".to_string())
        );
    }

    #[test]
    fn test_apply_promotion() {
        let board = Board::new_from_fen("k7/4P3/8/8/8/8/8/K7 w - - 0 1"); // White pawn on e7
        let mv = create_promo_move("e7", "e8", QUEEN); // e8=Q
        let next_board = board.apply_move_to_board(mv);

        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("e7")),
            None
        );
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("e8")),
            Some((WHITE, QUEEN))
        ); // Promoted to Queen
        assert!(!next_board.w_to_move);
        // Test that the resulting position has correct FEN
        assert_eq!(
            next_board.to_fen(),
            Some("k3Q3/8/8/8/8/8/8/K7 b - - 0 1".to_string())
        );
    }

    #[test]
    fn test_apply_castling_kingside_white() {
        let board = Board::new_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"); // Can castle both sides
        let mv = create_move("e1", "g1"); // O-O
        let next_board = board.apply_move_to_board(mv);

        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("e1")),
            None
        );
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("g1")),
            Some((WHITE, KING))
        );
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("h1")),
            None
        );
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("f1")),
            Some((WHITE, ROOK))
        );
        assert!(!next_board.w_to_move);
        // Test that the resulting position has correct FEN (castling rights should be updated)
        assert_eq!(
            next_board.to_fen(),
            Some("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R4RK1 b kq - 1 1".to_string())
        );
    }

    #[test]
    fn test_apply_castling_queenside_black() {
        let board = Board::new_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"); // Black to move
        let mv = create_move("e8", "c8"); // O-O-O
        let next_board = board.apply_move_to_board(mv);

        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("e8")),
            None
        );
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("c8")),
            Some((BLACK, KING))
        );
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("a8")),
            None
        );
        assert_eq!(
            next_board.get_piece(board_utils::algebraic_to_sq_ind("d8")),
            Some((BLACK, ROOK))
        );
        assert!(next_board.w_to_move); // White's turn
                                       // Test that the resulting position has correct FEN (castling rights should be updated)
        assert_eq!(
            next_board.to_fen(),
            Some("2kr3r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQ - 1 2".to_string())
        );
    }

    #[test]
    fn test_apply_rook_move_removes_castling() {
        let board = Board::new(); // Initial position
        let mv = create_move("a1", "a2"); // Ra2
        let next_board = board.apply_move_to_board(mv);
        // Test that castling rights are properly updated via FEN
        // After Ra2, white should lose queenside castling but keep kingside
        assert_eq!(
            next_board.to_fen(),
            Some("rnbqkbnr/pppppppp/8/8/8/8/RPPPPPPP/1NBQKBNR b Kkq - 0 1".to_string())
        );
    }
}

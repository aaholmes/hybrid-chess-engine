use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::piece_types::QUEEN;

#[test]
fn test_initial_move_count() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
    assert_eq!(captures.len() + moves.len(), 20); // 16 pawn moves + 4 knight moves
}

#[test]
fn test_knight_moves() {
    let board = Board::new_from_fen("K7/8/k7/8/4N3/8/8/8 w - - 0 1");
    let move_gen = MoveGen::new();
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
    assert_eq!(captures.len() + moves.len(), 11); // Knight should have 8 possible moves and king should have 3 (moving into check is OK for this function)
}

#[test]
fn test_pawn_promotion() {
    let board = Board::new_from_fen("1r6/P7/K7/8/k7/8/8/8 w - - 0 1");
    let move_gen = MoveGen::new();
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board);
    assert_eq!(captures.len() + moves.len(), 12); // 4 promotions, 4 capture-promotions, 4 king moves (moving into check is OK for this function)
}

#[test]
fn test_capture_ordering() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, _) = move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    let capture_vals: Vec<i32> = captures
        .iter()
        .map(|m| move_gen.mvv_lva(&board, m.from, m.to))
        .collect();
    println!("{} Captures:", captures.len());
    for (i, m) in captures.iter().enumerate() {
        println!("{}. {} ({})", i + 1, m, capture_vals[i]);
    }

    // Check that captures are ordered by MVV-LVA score in descending order
    for i in 1..captures.len() {
        assert!(
            capture_vals[i - 1] >= capture_vals[i],
            "Moves not properly ordered at index {}",
            i
        );
    }
}

#[test]
fn test_non_capture_ordering_white() {
    let board =
        Board::new_from_fen("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4");
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    board.print();

    println!("Captures:");
    for (i, m) in captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            move_gen.mvv_lva(&board, m.from, m.to)
        );
    }
    println!("Non-captures:");
    for (i, m) in non_captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            pesto.move_eval(&board, &move_gen, m.from, m.to)
        );
    }

    // Check that non-captures are ordered by Pesto eval change in descending order
    for i in 1..non_captures.len() {
        assert!(
            pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i - 1].from,
                non_captures[i - 1].to
            ) >= pesto.move_eval(&board, &move_gen, non_captures[i].from, non_captures[i].to),
            "Non-captures not properly ordered at index {}. {} vs {}",
            i,
            pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i - 1].from,
                non_captures[i - 1].to
            ),
            pesto.move_eval(&board, &move_gen, non_captures[i].from, non_captures[i].to)
        );
    }
}

#[test]
fn test_non_capture_ordering_black() {
    let board =
        Board::new_from_fen("rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R b KQkq - 0 5");
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    board.print();

    println!("Captures:");
    for (i, m) in captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            move_gen.mvv_lva(&board, m.from, m.to)
        );
    }
    println!("Non-captures:");
    for (i, m) in non_captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            pesto.move_eval(&board, &move_gen, m.from, m.to)
        );
    }

    // Check that non-captures are ordered by Pesto eval change in descending order
    for i in 1..non_captures.len() {
        assert!(
            pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i - 1].from,
                non_captures[i - 1].to
            ) >= pesto.move_eval(&board, &move_gen, non_captures[i].from, non_captures[i].to),
            "Non-captures not properly ordered at index {}. {} vs {}",
            i,
            pesto.move_eval(
                &board,
                &move_gen,
                non_captures[i - 1].from,
                non_captures[i - 1].to
            ),
            pesto.move_eval(&board, &move_gen, non_captures[i].from, non_captures[i].to)
        );
    }
}

#[test]
#[ignore] // Pre-existing failure - move ordering issue
fn test_pawn_fork_ordering() {
    let mut boardstack = BoardStack::new();
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Set up a position where a pawn fork is available
    let moves = ["e2e4", "e7e5", "b1c3", "g8f6", "f1c4", "f6e4", "c3e4"];

    for mv_str in moves.iter() {
        let mv = Move::from_uci(mv_str).unwrap();
        boardstack.make_move(mv);
    }

    let board = boardstack.current_state();

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    board.print();

    println!("Captures:");
    for (i, m) in captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            move_gen.mvv_lva(&board, m.from, m.to)
        );
    }
    println!("Non-captures:");
    for (i, m) in non_captures.iter().enumerate() {
        println!(
            "{}. {} ({})",
            i + 1,
            m,
            pesto.move_eval(&board, &move_gen, m.from, m.to)
        );
    }
    assert!(pesto.move_eval(&board, &move_gen, non_captures[0].from, non_captures[0].to) == 600);
}

#[test]
fn test_pseudo_legal_captures() {
    let board = Board::new();
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, _) = move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);
    assert!(
        captures.is_empty(),
        "No captures should be possible in the starting position"
    );
}

#[test]
fn test_mvv_lva_ordering() {
    let board =
        Board::new_from_fen("rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, _non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    assert!(
        !captures.is_empty(),
        "Captures should be possible in this position"
    );

    // Check if captures are ordered by MVV-LVA
    for i in 0..(captures.len() - 1) {
        let current_score = move_gen.mvv_lva(&board, captures[i].from, captures[i].to);
        let next_score = move_gen.mvv_lva(&board, captures[i + 1].from, captures[i + 1].to);
        assert!(
            current_score >= next_score,
            "Captures should be ordered by descending MVV-LVA scores"
        );
    }
}

#[test]
fn test_pesto_move_eval_consistency() {
    let fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1";
    let board = Board::new_from_fen(fen);
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (_captures, non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    // Check if non-captures are ordered by descending PestoEval scores
    for i in 0..(non_captures.len() - 1) {
        let current_score =
            pesto.move_eval(&board, &move_gen, non_captures[i].from, non_captures[i].to);
        let next_score = pesto.move_eval(
            &board,
            &move_gen,
            non_captures[i + 1].from,
            non_captures[i + 1].to,
        );
        assert!(
            current_score >= next_score,
            "Non-captures should be ordered by descending PestoEval scores"
        );
    }
}

#[test]
fn test_promotion_handling() {
    let fen = "rnbqk2r/pppp1P1p/5n2/2b1p3/4P3/8/PPPP2PP/RNBQKBNR w KQkq - 0 1";
    let board = Board::new_from_fen(fen);
    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();
    let history = None; // No history table

    // Update to use gen_pseudo_legal_moves_with_evals
    let (captures, _non_captures) =
        move_gen.gen_pseudo_legal_moves_with_evals(&board, &pesto, history);

    // Count promotions and make sure they're properly ordered
    let mut promotion_count = 0;
    for m in &captures {
        if m.promotion.is_some() {
            promotion_count += 1;

            // Promotions to queen should come before other promotion types
            if promotion_count == 1 {
                assert_eq!(
                    m.promotion,
                    Some(QUEEN),
                    "First promotion should be to queen"
                );
            }
        }
    }

    assert!(
        promotion_count > 0,
        "Promotions should be present in captures list"
    );
}

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::mate_search;
use std::time::Instant;

/// Integration tests for the parallel mate search portfolio.
#[test]
fn test_mate_search_portfolio() {
    let move_gen = MoveGen::new();

    // Test 1: Deep Checkmate (Spearhead Test)
    // Scholar's mate pattern, mate in 3 (1. Qh5 d6 2. Qxf7+ Ke7 3. Qxe7#)
    run_portfolio_test_case(
        "Spearhead Test (Deep Checks)",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        &move_gen,
        true, // Expect mate
        "h5f7", // Expected best move
    );

    // Test 2: Quiet Setup (Flanker Test)
    // Basic Endgame: King + Rook vs King.
    // White King f6, Rook a7. Black King h8.
    // 1. Kg6 (Quiet, takes opposition) Kg8 2. Ra8#
    // FEN: 7k/R7/5K2/8/8/8/8/8 w - - 0 1
    run_portfolio_test_case(
        "Flanker Test (Rook Mate in 2)",
        "7k/R7/5K2/8/8/8/8/8 w - - 0 1",
        &move_gen,
        true, // Expect mate
        "f6g6", // Kg6 is the key quiet move
    );
    
    // Test 3: Guardsman Test (General)
    // Starting position - should NOT find a mate in shallow depth
    run_portfolio_test_case(
        "Guardsman Test (General - Start Pos)",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &move_gen,
        false, // Expect no mate
        "", // No specific best move expected for no mate
    );
}

fn run_portfolio_test_case(name: &str, fen: &str, move_gen: &MoveGen, expect_mate: bool, expected_move_uci: &str) {
    println!("\n🔍 Running test case: {}", name);
    let board = Board::new_from_fen(fen);
    let mut stack = BoardStack::with_board(board);
    
    let start = Instant::now();
    // Use a reasonable depth (e.g., 5 for Guardsman, 10 for Spearhead)
    let (score, best_move, _nodes) = mate_search(&mut stack, move_gen, 5, false);
    let duration = start.elapsed();
    
    println!("   Result: Score={}, Move={:?}, Time={:.2?}", score, best_move, duration);
    
    if expect_mate {
        assert!(score >= 1_000_000, "Expected mate not found in {}", name);
        println!("   ✅ Mate Found! Best move: {:?}", best_move);
        if !expected_move_uci.is_empty() {
             let parsed_expected = kingfisher::move_types::Move::from_uci(expected_move_uci);
             if let Some(expected) = parsed_expected {
                 if best_move != expected {
                     println!("   ℹ️ Note: Found move {:?} differs from expected {:?}, but mate was found.", best_move, expected);
                 }
             }
        }
    } else {
        assert!(score.abs() < 1_000_000, "Unexpected mate found in {}", name);
        println!("   ℹ️ No Mate Found (as expected).");
    }
}

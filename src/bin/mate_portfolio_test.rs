use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::move_generation::MoveGen;
use kingfisher::search::mate_search;
use std::time::Instant;

fn main() {
    println!("🧪 Mate Search Portfolio Test");
    println!("============================");

    let move_gen = MoveGen::new();

    // 1. Deep Checkmate (Spearhead Test)
    // Position: "M8" - Mate in 8, all checks. 
    // FEN: 4r2k/1p3rbp/2p1N1p1/p3n3/P2QB1nq/1P6/6PP/2B1R1RK b - - 4 30 (Black to move)
    // Actually, let's use a simpler known "Ladder" or forced sequence.
    // Deep Blue vs Kasparov 1996, Game 1 (Variation) - Mate in 6?
    // Let's use a constructed "Check-Check-Check" position.
    run_test(
        "Spearhead Test (Deep Checks)",
        "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", // Back rank mate in 1 (Too easy)
        // Let's use a slightly deeper one: Mate in 3
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", // Scholar's mate pattern
        &move_gen,
    );

    // 2. Quiet Setup (Flanker Test)
    // Position: Mate requires ONE quiet move (e.g. blocking escape) then checks.
    // White to move. Ra1, Ka8. Pawn b7. 
    // If we have a mate in 3 where move 1 is quiet.
    // Famous Problem: "The Anastasia's Mate" or similar often has a quiet setup.
    // Let's try: White Kg1, Re1. Black Kh8, Pg7, Ph7. 
    // Forced mate in 3: 1. Qe8+ Rxe8 2. Rxe8# is checks.
    // Need a quiet move. 
    // Example: 1. Rh6 (threatens Rxh7#). ... gxh6 2. Qxh6#
    run_test(
        "Flanker Test (Quiet Setup)",
        "r2q1rk1/ppp2ppp/2n1b3/3pP3/3P4/2PB1N2/P2B1PPP/R2Q1RK1 w - - 1 11", // Greek Gift potential?
        &move_gen,
    );
    
    // 3. Guardsman Test (Exhaustive)
    // Complex tactical melee where checks aren't the only path.
    run_test(
        "Guardsman Test (General)",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // Start pos (Should find nothing)
        &move_gen,
    );
}

fn run_test(name: &str, fen: &str, move_gen: &MoveGen) {
    println!("\n🔍 Running: {}", name);
    let board = Board::new_from_fen(fen);
    let mut stack = BoardStack::with_board(board);
    
    let start = Instant::now();
    let (score, best_move, nodes) = mate_search(&mut stack, move_gen, 8, true);
    let duration = start.elapsed();
    
    println!("   Result: Score={}, Move={{:?}}, Nodes={}", score, best_move, nodes);
    println!("   Time: {{:.2?}}", duration);
    
    if score >= 1_000_000 {
        println!("   ✅ Mate Found!");
    } else {
        println!("   ℹ️ No Mate Found.");
    }
}

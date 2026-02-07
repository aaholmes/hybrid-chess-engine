use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search, TacticalMctsConfig, print_search_stats};
use kingfisher::neural_net::NeuralNetPolicy;
use std::time::Duration;

fn main() {
    println!("🎯 Testing Tactical-First MCTS Implementation");
    
    let board = Board::new(); // Starting position
    let move_gen = MoveGen::new();
    let mut nn_policy = None; // Use no neural network for now
    
    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_millis(500),
        mate_search_depth: 1, // Reduce mate search depth to minimize output
        exploration_constant: 1.414,
        use_neural_policy: false,
    };
    
    println!("🚀 Running tactical-first MCTS search...");
    let (best_move, stats, _) = tactical_mcts_search(
        board,
        &move_gen,
        &mut nn_policy,
        config,
    );
    
    print_search_stats(&stats, best_move);
    
    if best_move.is_some() {
        println!("✅ Tactical-first MCTS successfully found a move!");
    } else {
        println!("❌ No move found");
    }
    
    // Test with a tactical position
    println!("\n🎯 Testing with tactical position (scholar's mate setup):");
    let tactical_board = Board::new_from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3");
    
    let config2 = TacticalMctsConfig {
        max_iterations: 100,
        time_limit: Duration::from_millis(1000),
        mate_search_depth: 2, // Reduce mate search depth
        exploration_constant: 1.414,
        use_neural_policy: false,
    };
    
    let (best_move2, stats2, _) = tactical_mcts_search(
        tactical_board,
        &move_gen,
        &mut nn_policy,
        config2,
    );
    
    print_search_stats(&stats2, best_move2);
    
    if stats2.mates_found > 0 {
        println!("🏆 Tactical-first MCTS found mate sequences!");
    } else {
        println!("📈 Standard position evaluation completed");
    }
    
    println!("\n✅ Tactical-first MCTS implementation test completed successfully!");
}
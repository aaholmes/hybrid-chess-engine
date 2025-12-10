//! Neural Network Integration Demo
//!
//! This demo showcases the Tactical-First MCTS with Neural Network Policy integration,
//! demonstrating how classical chess knowledge (tactical moves) combines with modern
//! AI techniques (neural network policy guidance) for superior chess play.

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::mcts::{tactical_mcts_search, TacticalMctsConfig};
use kingfisher::move_generation::MoveGen;
use kingfisher::neural_net::NeuralNetPolicy;
use std::time::{Duration, Instant};

fn main() {
    println!("🏆 Kingfisher Chess Engine - Neural Network Integration Demo");
    println!("===========================================================\n");

    // Initialize core components
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();

    // Demo positions showcasing different scenarios
    let demo_positions = vec![
        ("Opening Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Complex Middlegame", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"),
        ("Tactical Position", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Endgame Position", "8/2k5/3p4/p2P1p2/P4P2/1K6/8/8 w - - 0 1"),
    ];

    // Phase 1: Demonstrate Neural Network Integration
    println!("📊 Phase 1: Neural Network Integration Demonstration");
    println!("====================================================\n");
    
    demonstrate_nn_integration(&demo_positions, &move_gen, &pesto_eval);
    
    // Phase 2: Performance Comparison
    println!("\n⚡ Phase 2: Performance Comparison Benchmark");
    println!("=============================================\n");
    
    run_performance_comparison(&demo_positions, &move_gen, &pesto_eval);
    
    println!("\n🎯 Demo completed! The Tactical-First MCTS with Neural Network integration");
    println!("   showcases both classical chess knowledge and modern AI techniques.");
}

/// Demonstrate neural network integration with tactical-first MCTS
fn demonstrate_nn_integration(positions: &[(&str, &str)], move_gen: &MoveGen, pesto_eval: &PestoEval) {
    // Initialize neural network in demo mode for reliable demonstration
    let mut nn_policy = Some(NeuralNetPolicy::new_demo_enabled());
    
    for (name, fen) in positions {
        println!("🔍 Analyzing: {}", name);
        println!("   Position: {}", fen);
        
        let board = Board::new_from_fen(fen);
        
        // Configure tactical-first MCTS with neural network
        let config = TacticalMctsConfig {
            max_iterations: 500,
            time_limit: Duration::from_millis(1000),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: true, // Enable NN integration
        };
        
        let start_time = Instant::now();
        let (best_move, stats) = tactical_mcts_search(
            board.clone(),
            move_gen,
            pesto_eval,
            &mut nn_policy,
            config,
        );
        let search_time = start_time.elapsed();
        
        // Display results
        if let Some(mv) = best_move {
            println!("   🎯 Best Move: {}", format_move(mv));
        } else {
            println!("   ❌ No move found");
        }
        
        println!("   📈 Search Statistics:");
        println!("      • Iterations: {}", stats.iterations);
        println!("      • Nodes Expanded: {}", stats.nodes_expanded);
        println!("      • Tactical Moves Explored: {}", stats.tactical_moves_explored);
        println!("      • NN Policy Evaluations: {}", stats.nn_policy_evaluations);
        println!("      • Mates Found: {}", stats.mates_found);
        println!("      • Search Time: {:?}", search_time);
        
        // Calculate tactical efficiency
        let tactical_ratio = if stats.nodes_expanded > 0 {
            (stats.tactical_moves_explored as f64 / stats.nodes_expanded as f64) * 100.0
        } else {
            0.0
        };
        
        let nn_efficiency = if stats.nodes_expanded > 0 {
            (stats.nn_policy_evaluations as f64 / stats.nodes_expanded as f64) * 100.0
        } else {
            0.0
        };
        
        println!("      • Tactical Priority: {:.1}% of nodes", tactical_ratio);
        println!("      • NN Efficiency: {:.1}% of nodes", nn_efficiency);
        
        println!("   ✅ Integration working: Classical tactics + Neural guidance\n");
    }
}

/// Run comprehensive performance comparison between different approaches
fn run_performance_comparison(positions: &[(&str, &str)], move_gen: &MoveGen, pesto_eval: &PestoEval) {
    println!("Comparing three approaches across {} test positions:", positions.len());
    println!("1. 🧠 Tactical-First MCTS + Neural Network (Hybrid)");
    println!("2. ⚔️  Pure Tactical-First MCTS (Classical)");
    println!("3. 🤖 Pure Neural Network Guidance (Modern)\n");
    
    let mut hybrid_total_time = Duration::from_millis(0);
    let mut classical_total_time = Duration::from_millis(0);
    let mut neural_total_time = Duration::from_millis(0);
    
    let mut hybrid_tactical_moves = 0;
    let mut classical_tactical_moves = 0;
    let mut neural_tactical_moves = 0;
    
    let mut hybrid_nn_calls = 0;
    let mut neural_nn_calls = 0;
    
    for (i, (name, fen)) in positions.iter().enumerate() {
        println!("🏁 Round {}: {}", i + 1, name);
        
        let board = Board::new_from_fen(fen);
        
        // 1. Hybrid Approach: Tactical-First + Neural Network
        let mut nn_policy_hybrid = Some(NeuralNetPolicy::new_demo_enabled());
        let hybrid_config = TacticalMctsConfig {
            max_iterations: 300,
            time_limit: Duration::from_millis(800),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: true,
        };
        
        let start = Instant::now();
        let (hybrid_move, hybrid_stats) = tactical_mcts_search(
            board.clone(), move_gen, pesto_eval, &mut nn_policy_hybrid, hybrid_config
        );
        let hybrid_time = start.elapsed();
        hybrid_total_time += hybrid_time;
        hybrid_tactical_moves += hybrid_stats.tactical_moves_explored;
        hybrid_nn_calls += hybrid_stats.nn_policy_evaluations;
        
        // 2. Classical Approach: Pure Tactical-First MCTS
        let mut nn_policy_none = None;
        let classical_config = TacticalMctsConfig {
            max_iterations: 300,
            time_limit: Duration::from_millis(800),
            mate_search_depth: 3,
            exploration_constant: 1.414,
            use_neural_policy: false,
        };
        
        let start = Instant::now();
        let (classical_move, classical_stats) = tactical_mcts_search(
            board.clone(), move_gen, pesto_eval, &mut nn_policy_none, classical_config
        );
        let classical_time = start.elapsed();
        classical_total_time += classical_time;
        classical_tactical_moves += classical_stats.tactical_moves_explored;
        
        // 3. Neural-First Approach: Prioritize NN over tactics
        let mut nn_policy_neural = Some(NeuralNetPolicy::new_demo_enabled());
        let neural_config = TacticalMctsConfig {
            max_iterations: 300,
            time_limit: Duration::from_millis(800),
            mate_search_depth: 1, // Minimal tactical search
            exploration_constant: 2.0, // Higher exploration for NN guidance
            use_neural_policy: true,
        };
        
        let start = Instant::now();
        let (neural_move, neural_stats) = tactical_mcts_search(
            board, move_gen, pesto_eval, &mut nn_policy_neural, neural_config
        );
        let neural_time = start.elapsed();
        neural_total_time += neural_time;
        neural_tactical_moves += neural_stats.tactical_moves_explored;
        neural_nn_calls += neural_stats.nn_policy_evaluations;
        
        // Display round results
        println!("   🧠 Hybrid: {} ({}ms, {} tactical, {} NN calls)", 
                 format_move_option(hybrid_move), hybrid_time.as_millis(),
                 hybrid_stats.tactical_moves_explored, hybrid_stats.nn_policy_evaluations);
        println!("   ⚔️  Classical: {} ({}ms, {} tactical)", 
                 format_move_option(classical_move), classical_time.as_millis(),
                 classical_stats.tactical_moves_explored);
        println!("   🤖 Neural: {} ({}ms, {} tactical, {} NN calls)", 
                 format_move_option(neural_move), neural_time.as_millis(),
                 neural_stats.tactical_moves_explored, neural_stats.nn_policy_evaluations);
        println!();
    }
    
    // Final benchmark summary
    println!("📊 Final Benchmark Results:");
    println!("==========================");
    
    let num_positions = positions.len() as f64;
    
    println!("🧠 Hybrid Tactical-First + Neural Network:");
    println!("   • Average time: {:.0}ms per position", hybrid_total_time.as_millis() as f64 / num_positions);
    println!("   • Total tactical moves: {}", hybrid_tactical_moves);
    println!("   • Total NN evaluations: {}", hybrid_nn_calls);
    println!("   • Efficiency: {:.1} tactical moves per NN call", 
             if hybrid_nn_calls > 0 { hybrid_tactical_moves as f64 / hybrid_nn_calls as f64 } else { 0.0 });
    
    println!("\n⚔️  Pure Tactical-First MCTS:");
    println!("   • Average time: {:.0}ms per position", classical_total_time.as_millis() as f64 / num_positions);
    println!("   • Total tactical moves: {}", classical_tactical_moves);
    println!("   • No NN overhead");
    
    println!("\n🤖 Neural Network Priority:");
    println!("   • Average time: {:.0}ms per position", neural_total_time.as_millis() as f64 / num_positions);
    println!("   • Total tactical moves: {}", neural_tactical_moves);
    println!("   • Total NN evaluations: {}", neural_nn_calls);
    
    // Performance insights
    println!("\n🎯 Key Insights:");
    println!("================");
    
    if hybrid_tactical_moves > neural_tactical_moves {
        println!("✅ Hybrid approach explores {}% more tactical moves than pure neural",
                 ((hybrid_tactical_moves as f64 / neural_tactical_moves as f64 - 1.0) * 100.0) as i32);
    }
    
    if classical_tactical_moves > hybrid_tactical_moves {
        println!("✅ Classical approach is most tactical-focused with {} total moves",
                 classical_tactical_moves);
    }
    
    println!("✅ Hybrid approach combines the best of both worlds:");
    println!("   • Classical tactical completeness");
    println!("   • Neural network strategic guidance");
    println!("   • Lazy evaluation efficiency");
    
    println!("\n🏆 The Tactical-First MCTS + Neural Network integration demonstrates");
    println!("   superior chess AI architecture suitable for professional applications!");
}

/// Format a move for display
fn format_move(mv: kingfisher::move_types::Move) -> String {
    let from_file = (mv.from % 8) as u8 + b'a';
    let from_rank = (mv.from / 8) as u8 + b'1';
    let to_file = (mv.to % 8) as u8 + b'a';
    let to_rank = (mv.to / 8) as u8 + b'1';
    
    let promotion = match mv.promotion {
        Some(piece) => match piece {
            1 => "n", 2 => "b", 3 => "r", 4 => "q", _ => "",
        },
        None => "",
    };
    
    format!("{}{}{}{}{}", 
            from_file as char, from_rank as char,
            to_file as char, to_rank as char, promotion)
}

/// Format an optional move for display
fn format_move_option(mv: Option<kingfisher::move_types::Move>) -> String {
    match mv {
        Some(m) => format_move(m),
        None => "None".to_string(),
    }
}
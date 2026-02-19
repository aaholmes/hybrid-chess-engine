//! Profiling Binary for MCTS Operation Timing
//!
//! Runs N self-play games and reports aggregate timing statistics for each
//! operation in evaluate_leaf_node: KOTH-in-3, mate search, Q-search, and NN inference.

use kingfisher::boardstack::BoardStack;
use kingfisher::mcts::{
    reuse_subtree, tactical_mcts_search_for_training_with_reuse, TacticalMctsConfig,
    TimingAccumulator,
};
use kingfisher::move_generation::MoveGen;
use kingfisher::transposition::TranspositionTable;
use std::time::{Duration, Instant};

#[cfg(feature = "neural")]
use kingfisher::mcts::InferenceServer;
#[cfg(feature = "neural")]
use kingfisher::neural_net::NeuralNetPolicy;
#[cfg(feature = "neural")]
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_path: Option<String> = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1).cloned());

    let num_games: usize = args
        .iter()
        .position(|a| a == "--games")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let simulations: u32 = args
        .iter()
        .position(|a| a == "--simulations")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);

    let enable_koth = args.iter().any(|a| a == "--koth");
    let disable_tier1 = args.iter().any(|a| a == "--disable-tier1");
    let disable_material = args.iter().any(|a| a == "--disable-material");

    let batch_size: usize = args
        .iter()
        .position(|a| a == "--batch-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    println!("=== MCTS Operation Profiler ===");
    println!("  Games: {}", num_games);
    println!("  Simulations/move: {}", simulations);
    println!("  KOTH: {}", enable_koth);
    println!("  Tier1: {}", !disable_tier1);
    println!("  Material: {}", !disable_material);
    println!("  Model: {:?}", model_path);
    println!("  Batch size: {}", batch_size);
    println!();

    // Set up inference server if model provided
    #[cfg(feature = "neural")]
    let inference_server: Option<Arc<InferenceServer>> = if let Some(ref path) = model_path {
        let mut nn = NeuralNetPolicy::new();
        if let Err(e) = nn.load(path) {
            eprintln!("Failed to load model from {}: {}", path, e);
            None
        } else {
            println!("  Loaded model, batch_size={}", batch_size);
            Some(Arc::new(InferenceServer::new(nn, batch_size)))
        }
    } else {
        None
    };

    #[cfg(not(feature = "neural"))]
    let inference_server: Option<()> = None;

    let has_nn = inference_server.is_some();

    // Accumulators across all games
    let mut koth_acc = TimingAccumulator::default();
    let mut mate_acc = TimingAccumulator::default();
    let mut qsearch_acc = TimingAccumulator::default();
    let mut nn_acc = TimingAccumulator::default();
    let mut total_moves: u64 = 0;
    let mut total_iterations: u64 = 0;

    let overall_start = Instant::now();

    for game_idx in 0..num_games {
        let move_gen = MoveGen::new();

        let config = TacticalMctsConfig {
            max_iterations: simulations,
            time_limit: Duration::from_secs(300),
            mate_search_depth: if disable_tier1 { 0 } else { 5 },
            exploration_constant: 1.414,
            use_neural_policy: has_nn,
            #[cfg(feature = "neural")]
            inference_server: inference_server.clone(),
            #[cfg(not(feature = "neural"))]
            inference_server: None,
            logger: None,
            enable_koth,
            enable_tier1_gate: !disable_tier1,
            enable_material_value: !disable_material,
            enable_tier3_neural: has_nn,
            randomize_move_order: false,
            ..Default::default()
        };

        let mut board_stack = BoardStack::new();
        let mut tt = TranspositionTable::new();
        let mut previous_root = None;
        let mut move_count = 0u32;

        loop {
            let board = board_stack.current_state().clone();

            let result = tactical_mcts_search_for_training_with_reuse(
                board.clone(),
                &move_gen,
                config.clone(),
                previous_root.take(),
                &mut tt,
            );

            if result.best_move.is_none() {
                break;
            }

            // Merge timing stats from this search
            koth_acc.merge(&result.stats.koth_timing);
            mate_acc.merge(&result.stats.mate_search_timing);
            qsearch_acc.merge(&result.stats.qsearch_timing);
            nn_acc.merge(&result.stats.nn_timing);
            total_iterations += result.stats.iterations as u64;

            let selected_move = result.best_move.unwrap();
            previous_root = reuse_subtree(result.root_node, selected_move);

            board_stack.make_move(selected_move);
            move_count += 1;

            // Check termination conditions
            if enable_koth {
                let (wk, bk) = board_stack.current_state().is_koth_win();
                if wk || bk {
                    break;
                }
            }
            if board_stack.is_draw_by_repetition() {
                break;
            }
            if board_stack.current_state().halfmove_clock() >= 100 {
                break;
            }
            if move_count > 200 {
                break;
            }
            let (mate, stalemate) = board_stack
                .current_state()
                .is_checkmate_or_stalemate(&move_gen);
            if mate || stalemate {
                break;
            }
        }

        total_moves += move_count as u64;
        println!(
            "  Game {}/{}: {} moves, {} iterations",
            game_idx + 1,
            num_games,
            move_count,
            total_iterations
        );
    }

    let overall_elapsed = overall_start.elapsed();

    // Print results
    println!();
    println!(
        "=== Profiling Results ({} games, {} total moves, {} total iterations) ===",
        num_games, total_moves, total_iterations
    );
    println!(
        "Total wall time: {:.1}s",
        overall_elapsed.as_secs_f64()
    );
    println!();
    println!(
        "{:<20} {:>10} {:>12} {:>12} {:>12}",
        "Operation", "Count", "Total(ms)", "Mean(us)", "Std(us)"
    );
    println!("{}", "-".repeat(68));

    print_row("KOTH-in-3", &koth_acc);
    print_row("Mate search", &mate_acc);
    print_row("Q-search", &qsearch_acc);
    print_row("NN inference", &nn_acc);
}

fn print_row(name: &str, acc: &TimingAccumulator) {
    if acc.count == 0 {
        println!(
            "{:<20} {:>10} {:>12} {:>12} {:>12}",
            name, 0, "-", "-", "-"
        );
    } else {
        println!(
            "{:<20} {:>10} {:>12.1} {:>12.1} {:>12.1}",
            name,
            acc.count,
            acc.total.as_secs_f64() * 1000.0,
            acc.mean_us(),
            acc.std_us()
        );
    }
}

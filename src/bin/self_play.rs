//! Self-Play Data Generation Binary
//! 
//! This binary plays games of the engine against itself to generate training data
//! for the neural network. It outputs JSON files containing FENs, MCTS policies,
//! and game outcomes.

use kingfisher::board::Board;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::{tactical_mcts_search_for_training, TacticalMctsConfig};
use kingfisher::neural_net::NeuralNetPolicy;
use serde::Serialize;
use std::fs::File;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;

#[derive(Serialize)]
struct TrainingSample {
    fen: String,
    policy: Vec<(String, u32)>, // Move UCI -> Visit Count
    value_target: f32,          // +1 (White Win), -1 (Black Win), 0 (Draw)
    mcts_value: f64,            // Q-value from search (optional aux target)
}

#[derive(Serialize)]
struct GameRecord {
    samples: Vec<TrainingSample>,
    result: String, // "1-0", "0-1", "1/2-1/2"
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_games = if args.len() > 1 { args[1].parse().unwrap_or(1) } else { 1 };
    let simulations = if args.len() > 2 { args[2].parse().unwrap_or(800) } else { 800 };
    let output_dir = if args.len() > 3 { &args[3] } else { "data" };

    println!("🤖 Self-Play Generator Starting...");
    println!("   Games: {}", num_games);
    println!("   Simulations/Move: {}", simulations);
    println!("   Output Dir: {}", output_dir);

    std::fs::create_dir_all(output_dir).unwrap();

    let completed_games = Mutex::new(0);

    // Run games in parallel
    (0..num_games).into_par_iter().for_each(|i| {
        let game_data = play_game(i, simulations);
        
        // Save game data
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let filename = format!("{}/game_{}_{}.json", output_dir, timestamp, i);
        let file = File::create(&filename).unwrap();
        serde_json::to_writer(file, &game_data).unwrap();
        
        let mut count = completed_games.lock().unwrap();
        *count += 1;
        println!("✅ Game {}/{} finished. Result: {}", *count, num_games, game_data.result);
    });
}

fn play_game(game_num: usize, simulations: u32) -> GameRecord {
    let mut board = Board::new();
    let move_gen = MoveGen::new();
    let pesto_eval = PestoEval::new();
    
    // Each thread gets its own NN instance (if available)
    // Note: If using CUDA, multiple threads might contend.
    // Ideally, we'd use a shared Batching inference service, but simple separate instances work for now.
    let mut nn_policy = Some(NeuralNetPolicy::new_demo_enabled()); 

    let mut samples = Vec::new();
    let mut move_count = 0;
    
    loop {
        // 1. MCTS Search
        let config = TacticalMctsConfig {
            max_iterations: simulations,
            time_limit: Duration::from_secs(60), // Time shouldn't be the limit, sims should
            mate_search_depth: 1, // Reduced depth for speed during self-play
            exploration_constant: 1.414,
            use_neural_policy: true, // Use the NN we loaded
        };

        let result = tactical_mcts_search_for_training(
            board.clone(),
            &move_gen,
            &pesto_eval,
            &mut nn_policy,
            config,
        );

        if result.best_move.is_none() {
            break; // Game Over
        }

        // 2. Store Sample (Outcome unknown yet)
        let fen = board.to_fen().unwrap_or_default();
        let policy: Vec<(String, u32)> = result.root_policy.iter()
            .map(|(m, visits)| (m.to_uci(), *visits))
            .collect();
        
        samples.push(TrainingSample {
            fen: fen.clone(),
            policy,
            value_target: 0.0, // Placeholder
            mcts_value: result.root_value_prediction,
        });

        // 3. Play Move
        // Temperature logic: First 30 moves proportional to visits, then max visits.
        // For simplicity in this v1, let's just pick best move (max visits).
        // Or simple temperature:
        let selected_move = result.best_move.unwrap(); 
        
        // Print progress for this game instance
        println!("  Game {}: Move {} - Playing {} from FEN: {}", 
                 game_num, move_count + 1, selected_move.to_uci(), fen);
        
        // Apply move
        board = board.apply_move_to_board(selected_move);
        move_count += 1;

        // Check for draw/end
        if move_count > 200 { // Draw by length
            break; 
        }
        
        // Is game over?
        let (mate, _stalemate) = board.is_checkmate_or_stalemate(&move_gen);
        if mate || _stalemate {
            break;
        }
    }

    // 4. Assign Outcomes
    let (mate, stalemate) = board.is_checkmate_or_stalemate(&move_gen);
    let result_str;
    let final_score_white; // 1.0 (W win), -1.0 (B win), 0.0 (Draw)

    if mate {
        // Side to move lost.
        if board.w_to_move {
            final_score_white = -1.0; // Black wins
            result_str = "0-1";
        } else {
            final_score_white = 1.0; // White wins
            result_str = "1-0";
        }
    } else {
        // Stalemate or Draw
        final_score_white = 0.0;
        result_str = "1/2-1/2";
    }

    // Backpropagate Z
    for (i, sample) in samples.iter_mut().enumerate() {
        // Sample stored FEN. Who was to move?
        // Game starts White. i=0 is White. i=1 is Black.
        let white_to_move_at_sample = i % 2 == 0;
        
        if white_to_move_at_sample {
            sample.value_target = final_score_white;
        } else {
            sample.value_target = -final_score_white;
        }
    }

    GameRecord {
        samples,
        result: result_str.to_string(),
    }
}

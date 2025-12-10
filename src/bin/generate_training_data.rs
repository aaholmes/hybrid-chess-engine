//! Training Data Generation Binary
//!
//! Generates training data for neural network policy by analyzing positions
//! and creating datasets suitable for PyTorch training.

use kingfisher::training::{TrainingDataGenerator, ParsedGame};
use kingfisher::move_types::Move;

fn main() {
    println!("🎓 Kingfisher Training Data Generator");
    println!("=====================================");
    
    let mut generator = TrainingDataGenerator::new();
    generator.set_search_depth(6); // Reasonable depth for training data analysis
    
    // Generate tactical training positions
    println!("\n📊 Generating tactical positions...");
    let tactical_positions = generator.generate_tactical_positions();
    println!("✅ Generated {} tactical positions", tactical_positions.len());
    
    // Create some sample games for demonstration
    println!("\n🎮 Generating sample game positions...");
    let sample_games = create_sample_games();
    let game_positions = generator.generate_from_games(&sample_games);
    println!("✅ Generated {} game positions", game_positions.len());
    
    // Combine all positions
    let mut all_positions = tactical_positions;
    all_positions.extend(game_positions);
    
    println!("\n💾 Total training positions: {}", all_positions.len());
    
    // Save to CSV file for Python training pipeline
    let csv_path = "training_data.csv";
    match generator.save_to_csv(&all_positions, csv_path) {
        Ok(()) => println!("✅ Saved training data to: {}", csv_path),
        Err(e) => println!("❌ Failed to save CSV: {}", e),
    }
    
    // Test loading the data back
    println!("\n🔄 Testing data loading...");
    match TrainingDataGenerator::load_from_csv(csv_path) {
        Ok(loaded_positions) => {
            println!("✅ Successfully loaded {} positions", loaded_positions.len());
            
            // Show some statistics
            let white_wins = loaded_positions.iter().filter(|p| p.game_result > 0.7).count();
            let draws = loaded_positions.iter().filter(|p| (p.game_result - 0.5).abs() < 0.2).count();
            let black_wins = loaded_positions.iter().filter(|p| p.game_result < 0.3).count();
            
            println!("📈 Dataset statistics:");
            println!("   White wins: {} ({:.1}%)", white_wins, white_wins as f64 / loaded_positions.len() as f64 * 100.0);
            println!("   Draws: {} ({:.1}%)", draws, draws as f64 / loaded_positions.len() as f64 * 100.0);
            println!("   Black wins: {} ({:.1}%)", black_wins, black_wins as f64 / loaded_positions.len() as f64 * 100.0);
            
            // Show sample positions
            println!("\n🔍 Sample positions:");
            for (i, pos) in loaded_positions.iter().take(3).enumerate() {
                println!("   {}. {} (result: {:.1}, eval: {:?})", 
                        i + 1, 
                        pos.description,
                        pos.game_result,
                        pos.engine_eval);
            }
        }
        Err(e) => println!("❌ Failed to load CSV: {}", e),
    }
    
    println!("\n🎉 Training data generation complete!");
    println!("\nNext steps:");
    println!("1. Use this data with the Python training pipeline:");
    println!("   python3 python/training_pipeline.py --csv training_data.csv");
    println!("2. Or collect larger datasets:");
    println!("   python3 python/data_collection.py --action sample");
    println!("3. Train the neural network:");
    println!("   python3 python/training_pipeline.py --pgn data/sample_games.pgn");
}

/// Create sample games for demonstration
fn create_sample_games() -> Vec<ParsedGame> {
    let mut games = Vec::new();
    
    // Sample game 1: Italian Game
    let mut game1 = ParsedGame::new();
    game1.result = 1.0; // White wins
    game1.white_elo = Some(2000);
    game1.black_elo = Some(1950);
    game1.metadata.insert("Event".to_string(), "Sample Game 1".to_string());
    
    // Add some moves (simplified for demonstration)
    game1.moves = vec![
        Move::from_uci("e2e4").unwrap(),
        Move::from_uci("e7e5").unwrap(),
        Move::from_uci("g1f3").unwrap(),
        Move::from_uci("b8c6").unwrap(),
        Move::from_uci("f1c4").unwrap(),
        Move::from_uci("f8c5").unwrap(),
        Move::from_uci("c2c3").unwrap(),
        Move::from_uci("g8f6").unwrap(),
        Move::from_uci("d2d4").unwrap(),
        Move::from_uci("e5d4").unwrap(),
        Move::from_uci("c3d4").unwrap(),
        Move::from_uci("c5b4").unwrap(),
        Move::from_uci("b1c3").unwrap(),
        Move::from_uci("f6e4").unwrap(),
        Move::from_uci("e1g1").unwrap(),
        Move::from_uci("b4c3").unwrap(),
        Move::from_uci("b2c3").unwrap(),
        Move::from_uci("d7d5").unwrap(),
        Move::from_uci("c4d3").unwrap(),
        Move::from_uci("e4d6").unwrap(),
    ];
    
    games.push(game1);
    
    // Sample game 2: Queen's Gambit Declined
    let mut game2 = ParsedGame::new();
    game2.result = 0.5; // Draw
    game2.white_elo = Some(1900);
    game2.black_elo = Some(1950);
    game2.metadata.insert("Event".to_string(), "Sample Game 2".to_string());
    
    game2.moves = vec![
        Move::from_uci("d2d4").unwrap(),
        Move::from_uci("d7d5").unwrap(),
        Move::from_uci("c2c4").unwrap(),
        Move::from_uci("e7e6").unwrap(),
        Move::from_uci("b1c3").unwrap(),
        Move::from_uci("g8f6").unwrap(),
        Move::from_uci("c1g5").unwrap(),
        Move::from_uci("f8e7").unwrap(),
        Move::from_uci("e2e3").unwrap(),
        Move::from_uci("e8g8").unwrap(),
        Move::from_uci("g1f3").unwrap(),
        Move::from_uci("b8d7").unwrap(),
        Move::from_uci("f1d3").unwrap(),
        Move::from_uci("c7c6").unwrap(),
        Move::from_uci("e1g1").unwrap(),
        Move::from_uci("d5c4").unwrap(),
        Move::from_uci("d3c4").unwrap(),
        Move::from_uci("f6d5").unwrap(),
    ];
    
    games.push(game2);
    
    // Sample game 3: Scandinavian Defense
    let mut game3 = ParsedGame::new();
    game3.result = 0.0; // Black wins
    game3.white_elo = Some(1800);
    game3.black_elo = Some(2100);
    game3.metadata.insert("Event".to_string(), "Sample Game 3".to_string());
    
    game3.moves = vec![
        Move::from_uci("e2e4").unwrap(),
        Move::from_uci("d7d5").unwrap(),
        Move::from_uci("e4d5").unwrap(),
        Move::from_uci("d8d5").unwrap(),
        Move::from_uci("b1c3").unwrap(),
        Move::from_uci("d5a5").unwrap(),
        Move::from_uci("d2d4").unwrap(),
        Move::from_uci("g8f6").unwrap(),
        Move::from_uci("g1f3").unwrap(),
        Move::from_uci("c8f5").unwrap(),
        Move::from_uci("f1d3").unwrap(),
        Move::from_uci("f5d3").unwrap(),
        Move::from_uci("d1d3").unwrap(),
        Move::from_uci("e7e6").unwrap(),
        Move::from_uci("c1f4").unwrap(),
        Move::from_uci("f8b4").unwrap(),
    ];
    
    games.push(game3);
    
    games
}
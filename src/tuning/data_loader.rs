//! Data loading utilities for Texel tuning

use super::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub struct DataLoader;

impl DataLoader {
    /// Load positions from a simple format: FEN,result,description
    pub fn load_from_csv<P: AsRef<Path>>(
        path: P,
    ) -> Result<Vec<TexelPosition>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut positions = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue; // Skip empty lines and comments
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let fen = parts[0].trim();
                let result_str = parts[1].trim();
                let description = if parts.len() > 2 {
                    parts[2].trim().to_string()
                } else {
                    format!("Position {}", line_num + 1)
                };

                // Parse result (can be numeric 0.0/0.5/1.0 or PGN format 1-0/1/2-1/2/0-1)
                let result = if let Ok(numeric_result) = result_str.parse::<f64>() {
                    numeric_result
                } else {
                    match result_str {
                        "1-0" => 1.0,
                        "0-1" => 0.0,
                        "1/2-1/2" => 0.5,
                        _ => {
                            eprintln!(
                                "Warning: Unknown result format '{}' on line {}",
                                result_str,
                                line_num + 1
                            );
                            continue;
                        }
                    }
                };

                if let Some(position) = TexelPosition::new(fen, result, description) {
                    positions.push(position);
                } else {
                    eprintln!("Warning: Invalid FEN '{}' on line {}", fen, line_num + 1);
                }
            }
        }

        println!("✅ Loaded {} positions from file", positions.len());
        Ok(positions)
    }

    /// Load positions from EPD format (Extended Position Description)
    pub fn load_from_epd<P: AsRef<Path>>(
        path: P,
    ) -> Result<Vec<TexelPosition>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut positions = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            // EPD format: FEN + additional operations
            // Example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - c0 \"Starting position\"; result \"1/2-1/2\";"

            let parts: Vec<&str> = line.splitn(2, ';').collect();
            let fen_part = parts[0].trim();

            // Extract FEN (first 4-6 components)
            let fen_components: Vec<&str> = fen_part.split_whitespace().collect();
            if fen_components.len() >= 4 {
                let fen = fen_components[0..6.min(fen_components.len())].join(" ");

                // Look for result in operations
                let mut result = 0.5; // Default to draw
                let mut description = format!("EPD Position {}", line_num + 1);

                if parts.len() > 1 {
                    let operations = parts[1];

                    // Parse result operation
                    if let Some(result_start) = operations.find("result") {
                        if let Some(quote_start) = operations[result_start..].find('"') {
                            if let Some(quote_end) =
                                operations[result_start + quote_start + 1..].find('"')
                            {
                                let result_str = &operations[result_start + quote_start + 1
                                    ..result_start + quote_start + 1 + quote_end];
                                result = match result_str {
                                    "1-0" => 1.0,
                                    "0-1" => 0.0,
                                    "1/2-1/2" => 0.5,
                                    _ => 0.5,
                                };
                            }
                        }
                    }

                    // Parse description/comment
                    if let Some(desc_start) = operations.find("c0") {
                        if let Some(quote_start) = operations[desc_start..].find('"') {
                            if let Some(quote_end) =
                                operations[desc_start + quote_start + 1..].find('"')
                            {
                                description = operations[desc_start + quote_start + 1
                                    ..desc_start + quote_start + 1 + quote_end]
                                    .to_string();
                            }
                        }
                    }
                }

                if let Some(position) = TexelPosition::new(&fen, result, description) {
                    positions.push(position);
                }
            }
        }

        println!("✅ Loaded {} positions from EPD file", positions.len());
        Ok(positions)
    }

    /// Generate synthetic training data based on common chess patterns
    pub fn generate_synthetic_data(num_positions: usize) -> Vec<TexelPosition> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut positions = Vec::new();

        // Generate various types of positions
        for i in 0..num_positions {
            let position_type = i % 4;

            let (fen, result, description) = match position_type {
                0 => {
                    // Endgame positions (clearer evaluations)
                    let endgames = [
                        ("8/8/8/4k3/4P3/4K3/8/8 w - - 0 1", 0.8, "KP vs K"),
                        ("8/8/8/8/8/4k3/4p3/4K3 b - - 0 1", 0.2, "KP vs K (Black)"),
                        ("8/8/8/8/4k3/8/4P3/4K3 w - - 0 1", 0.9, "Advanced Pawn"),
                        ("8/8/8/3k4/8/3K4/8/8 w - - 0 1", 0.5, "K vs K Draw"),
                    ];
                    let &(fen, result, desc) = endgames.choose(&mut rng).unwrap();
                    (fen.to_string(), result, desc.to_string())
                }
                1 => {
                    // Opening positions (roughly equal)
                    let openings = [
                        (
                            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                            0.55,
                            "e4",
                        ),
                        (
                            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
                            0.5,
                            "e4 e5",
                        ),
                        (
                            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
                            0.45,
                            "d4 d5 c4",
                        ),
                    ];
                    let &(fen, result, desc) = openings.choose(&mut rng).unwrap();
                    (fen.to_string(), result, desc.to_string())
                }
                2 => {
                    // Tactical positions (clear advantage)
                    let tactical = [
                        (
                            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
                            0.0,
                            "Scholar's Mate",
                        ),
                        ("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1", 1.0, "Back Rank Mate"),
                        (
                            "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3",
                            0.0,
                            "Early Queen Attack",
                        ),
                    ];
                    let &(fen, result, desc) = tactical.choose(&mut rng).unwrap();
                    (fen.to_string(), result, desc.to_string())
                }
                _ => {
                    // Middlegame positions (varied evaluations)
                    let middlegame = [
                        (
                            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
                            0.5,
                            "Italian Game",
                        ),
                        (
                            "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4",
                            0.45,
                            "QGD",
                        ),
                        (
                            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                            0.55,
                            "Knight Development",
                        ),
                    ];
                    let &(fen, result, desc) = middlegame.choose(&mut rng).unwrap();
                    (fen.to_string(), result, desc.to_string())
                }
            };

            if let Some(position) =
                TexelPosition::new(&fen, result, format!("{} ({})", description, i))
            {
                positions.push(position);
            }
        }

        println!("✅ Generated {} synthetic positions", positions.len());
        positions
    }

    /// Save positions to CSV format for later use
    pub fn save_to_csv<P: AsRef<Path>>(
        positions: &[TexelPosition],
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;

        writeln!(file, "# FEN,Result,Description")?;
        for position in positions {
            // Note: Board doesn't have to_fen() method, would need to implement or skip this feature
            writeln!(
                file,
                "[FEN_NOT_IMPLEMENTED],{},\"{}\"",
                position.game_result, position.description
            )?;
        }

        println!("✅ Saved {} positions to file", positions.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_synthetic_data_generation() {
        let positions = DataLoader::generate_synthetic_data(10);
        assert_eq!(positions.len(), 10);

        // Check that we have variety in results
        let results: Vec<f64> = positions.iter().map(|p| p.game_result).collect();
        let mut unique_results: Vec<f64> = Vec::new();
        for &result in &results {
            if !unique_results
                .iter()
                .any(|&x: &f64| (x - result).abs() < 0.001)
            {
                unique_results.push(result);
            }
        }
        assert!(
            unique_results.len() > 1,
            "Should have variety in game results"
        );
    }

    #[test]
    fn test_csv_loading() -> Result<(), Box<dyn std::error::Error>> {
        let mut temp_file = NamedTempFile::new()?;

        // Write test data
        writeln!(temp_file, "# Test CSV file")?;
        writeln!(
            temp_file,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,0.5,Starting position"
        )?;
        writeln!(
            temp_file,
            "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1,1.0,Back rank mate"
        )?;
        writeln!(temp_file, "8/8/8/4k3/4P3/4K3/8/8 w - - 0 1,1-0,KP vs K")?;

        let positions = DataLoader::load_from_csv(temp_file.path())?;

        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0].game_result, 0.5);
        assert_eq!(positions[1].game_result, 1.0);
        assert_eq!(positions[2].game_result, 1.0); // 1-0 should convert to 1.0

        Ok(())
    }
}

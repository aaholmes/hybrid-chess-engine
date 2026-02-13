//! Training data generation for neural network policy
//!
//! This module provides functionality to generate training data for the neural network
//! by analyzing positions with the engine and extracting features.

use crate::board::Board;
use crate::eval::PestoEval;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Training position with target values for neural network
#[derive(Debug, Clone)]
pub struct TrainingPosition {
    /// Board position as FEN string
    pub fen: String,
    /// Game result (1.0 = white wins, 0.5 = draw, 0.0 = black wins)
    pub game_result: f64,
    /// Best move in UCI format (if available)
    pub best_move: Option<String>,
    /// Engine evaluation in centipawns
    pub engine_eval: Option<i32>,
    /// Description/source of position
    pub description: String,
}

impl TrainingPosition {
    pub fn new(
        board: &Board,
        game_result: f64,
        best_move: Option<Move>,
        description: String,
    ) -> Self {
        TrainingPosition {
            fen: board.to_fen().unwrap_or_default(),
            game_result,
            best_move: best_move.map(|mv| mv.to_uci()),
            engine_eval: None,
            description,
        }
    }

    /// Convert to CSV format for Python training pipeline
    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},\"{}\"",
            self.fen,
            self.game_result,
            self.best_move.as_deref().unwrap_or(""),
            self.engine_eval.unwrap_or(0),
            self.description
        )
    }
}

/// Training data generator
pub struct TrainingDataGenerator {
    move_gen: MoveGen,
    pesto_eval: PestoEval,
    max_depth: i32,
}

impl TrainingDataGenerator {
    pub fn new() -> Self {
        TrainingDataGenerator {
            move_gen: MoveGen::new(),
            pesto_eval: PestoEval::new(),
            max_depth: 8,
        }
    }

    pub fn set_search_depth(&mut self, depth: i32) {
        self.max_depth = depth;
    }

    /// Generate training positions from a series of games
    pub fn generate_from_games(&self, games: &[ParsedGame]) -> Vec<TrainingPosition> {
        let mut positions = Vec::new();

        for game in games {
            let game_positions = self.extract_positions_from_game(game);
            positions.extend(game_positions);
        }

        println!(
            "Generated {} training positions from {} games",
            positions.len(),
            games.len()
        );
        positions
    }

    /// Extract training positions from a single game
    fn extract_positions_from_game(&self, game: &ParsedGame) -> Vec<TrainingPosition> {
        let mut positions = Vec::new();
        let mut board = Board::new();

        // Play through the game moves
        for (move_idx, mv) in game.moves.iter().enumerate() {
            // Only sample some positions (not every move)
            if move_idx % 3 == 0 && move_idx > 10 && move_idx < game.moves.len() - 10 {
                // This is a position we want to include in training
                if self.is_suitable_training_position(&board) {
                    let mut training_pos = TrainingPosition::new(
                        &board,
                        game.result,
                        Some(*mv),
                        format!("Game move {}", move_idx + 1),
                    );

                    // Analyze position with engine to get evaluation
                    if let Some(eval) = self.analyze_position(&board) {
                        training_pos.engine_eval = Some(eval);
                    }

                    positions.push(training_pos);
                }
            }

            // Apply the move to continue through the game
            let new_board = board.apply_move_to_board(*mv);
            if new_board.is_legal(&self.move_gen) {
                board = new_board;
            } else {
                break; // Invalid move, stop processing this game
            }
        }

        positions
    }

    /// Check if position is suitable for training
    fn is_suitable_training_position(&self, board: &Board) -> bool {
        // Skip positions in check (too tactical)
        if board.is_check(&self.move_gen) {
            return false;
        }

        // Skip positions with very few pieces
        let piece_count = self.count_pieces(board);
        if piece_count < 12 {
            return false;
        }

        // Skip positions that are likely drawn by repetition or 50-move rule
        if board.halfmove_clock > 40 {
            return false;
        }

        true
    }

    /// Count total pieces on board
    fn count_pieces(&self, board: &Board) -> u32 {
        let mut count = 0;
        for color in 0..2 {
            for piece in 0..6 {
                count += board.pieces[color][piece].count_ones();
            }
        }
        count
    }

    /// Analyze position with engine to get evaluation
    fn analyze_position(&self, board: &Board) -> Option<i32> {
        // Use Pesto evaluation for now (could be enhanced with search later)
        let eval = self.pesto_eval.eval(board, &self.move_gen);
        Some(eval)
    }

    /// Save training positions to CSV file
    pub fn save_to_csv<P: AsRef<Path>>(
        &self,
        positions: &[TrainingPosition],
        path: P,
    ) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Write header
        writeln!(file, "fen,result,best_move,engine_eval,description")?;

        // Write positions
        for position in positions {
            writeln!(file, "{}", position.to_csv_line())?;
        }

        println!("Saved {} positions to CSV", positions.len());
        Ok(())
    }

    /// Load training positions from CSV file
    pub fn load_from_csv<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<TrainingPosition>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut positions = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;

            // Skip header and empty lines
            if line_num == 0 || line.trim().is_empty() {
                continue;
            }

            if let Some(position) = Self::parse_csv_line(&line) {
                positions.push(position);
            }
        }

        println!("Loaded {} positions from CSV", positions.len());
        Ok(positions)
    }

    /// Parse a CSV line into a TrainingPosition
    fn parse_csv_line(line: &str) -> Option<TrainingPosition> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 5 {
            return None;
        }

        let fen = parts[0].to_string();
        let game_result = parts[1].parse().ok()?;
        let best_move = if parts[2].is_empty() {
            None
        } else {
            Some(parts[2].to_string())
        };
        let engine_eval = parts[3].parse().ok();
        let description = parts[4].trim_matches('"').to_string();

        Some(TrainingPosition {
            fen,
            game_result,
            best_move,
            engine_eval,
            description,
        })
    }

    /// Generate positions from tactical puzzles
    pub fn generate_tactical_positions(&self) -> Vec<TrainingPosition> {
        let mut positions = Vec::new();

        // Some well-known tactical positions
        let tactical_fens = vec![
            (
                "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1",
                1.0,
                "Re8#",
                "Back rank mate",
            ),
            (
                "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
                0.0,
                "",
                "Scholar's mate",
            ),
            (
                "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3",
                0.0,
                "",
                "Early queen attack",
            ),
            (
                "8/8/8/4k3/4P3/4K3/8/8 w - - 0 1",
                0.7,
                "Kd4",
                "King and pawn endgame",
            ),
            (
                "8/8/8/8/8/4k3/4p3/4K3 b - - 0 1",
                0.3,
                "Kd2",
                "King and pawn endgame (Black)",
            ),
        ];

        for (fen, result, best_move, description) in tactical_fens {
            let board = Board::new_from_fen(fen);
            let mv = if best_move.is_empty() {
                None
            } else {
                Move::from_uci(best_move)
            };

            let mut pos = TrainingPosition::new(&board, result, mv, description.to_string());
            if let Some(eval) = self.analyze_position(&board) {
                pos.engine_eval = Some(eval);
            }
            positions.push(pos);
        }

        positions
    }
}

/// Parsed game data
#[derive(Debug, Clone)]
pub struct ParsedGame {
    /// Game moves in order
    pub moves: Vec<Move>,
    /// Game result (1.0 = white wins, 0.5 = draw, 0.0 = black wins)
    pub result: f64,
    /// Player ratings
    pub white_elo: Option<u32>,
    pub black_elo: Option<u32>,
    /// Game metadata
    pub metadata: HashMap<String, String>,
}

impl ParsedGame {
    pub fn new() -> Self {
        ParsedGame {
            moves: Vec::new(),
            result: 0.5,
            white_elo: None,
            black_elo: None,
            metadata: HashMap::new(),
        }
    }
}

impl Default for TrainingDataGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ParsedGame {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_position_csv() {
        let board = Board::new();
        let mv = Move::new(12, 28, None); // e2e4
        let pos = TrainingPosition::new(&board, 0.6, Some(mv), "Test position".to_string());

        let csv_line = pos.to_csv_line();
        assert!(csv_line.contains("0.6"));
        assert!(csv_line.contains("Test position"));
    }

    #[test]
    fn test_tactical_positions() {
        let generator = TrainingDataGenerator::new();
        let positions = generator.generate_tactical_positions();

        assert!(!positions.is_empty());
        assert!(positions
            .iter()
            .any(|p| p.description.contains("Back rank")));
    }

    #[test]
    fn test_piece_counting() {
        let generator = TrainingDataGenerator::new();
        let board = Board::new();

        let count = generator.count_pieces(&board);
        assert_eq!(count, 32); // Starting position has 32 pieces
    }
}

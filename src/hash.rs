//! Zobrist Hashing Module
//!
//! This module implements Zobrist hashing for chess positions. Zobrist hashing
//! is a technique used to efficiently encode chess board states into unique
//! 64-bit integers. This is crucial for implementing features such as:
//!
//! - Detecting threefold repetition
//! - Implementing transposition tables
//! - Efficiently comparing board positions
//!
//! The module provides a global `ZOBRIST_KEYS` instance that should be used
//! throughout the chess engine to ensure consistency in hash generation.
//!
//! # Note
//!
//! The Zobrist keys are generated randomly at program startup and remain
//! constant for the duration of the program's execution. This ensures
//! that the same position will always hash to the same value within a
//! single run of the program.

use crate::bits::bits;
use crate::board::Board;
use crate::boardstack::BoardStack;
use crate::piece_types::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use lazy_static::lazy_static;
use rand::Rng;

const PIECE_TYPES: usize = 6; // Pawn, Knight, Bishop, Rook, Queen, King
const COLORS: usize = 2; // White, Black
const SQUARES: usize = 64;

/// Represents a set of Zobrist keys used for hashing chess positions.
///
/// These keys are used to create a unique hash for each chess position,
/// taking into account piece positions, castling rights, en passant possibilities,
/// and the side to move.
pub struct ZobristKeys {
    /// 3D array of keys for each piece type, color, and square
    piece_keys: [[[u64; SQUARES]; PIECE_TYPES]; COLORS],
    /// Keys for castling rights [KS_WHITE, QS_WHITE, KS_BLACK, QS_BLACK]
    castling_keys: [u64; 4],
    /// Keys for en passant possibilities on each file
    en_passant_keys: [u64; 8],
    /// Key to toggle for the side to move
    side_to_move_key: u64,
}

impl ZobristKeys {
    /// Get the Zobrist key for a piece at a specific square.
    pub fn piece_key(&self, color: usize, piece_type: usize, square: usize) -> u64 {
        self.piece_keys[color][piece_type][square]
    }

    /// Get the Zobrist key for a castling right.
    /// Index: 0=WK, 1=WQ, 2=BK, 3=BQ
    pub fn castling_key(&self, index: usize) -> u64 {
        self.castling_keys[index]
    }

    /// Get the Zobrist key for an en passant file.
    pub fn en_passant_key(&self, file: usize) -> u64 {
        self.en_passant_keys[file]
    }

    /// Get the Zobrist key for the side to move.
    pub fn side_to_move_key(&self) -> u64 {
        self.side_to_move_key
    }

    /// Generates a new set of random Zobrist keys.
    ///
    /// This method should typically only be called once to initialize the global ZOBRIST_KEYS.
    fn new() -> Self {
        let mut rng = rand::thread_rng();

        let mut keys = ZobristKeys {
            piece_keys: [[[0; SQUARES]; PIECE_TYPES]; COLORS],
            castling_keys: [0; 4],
            en_passant_keys: [0; 8],
            side_to_move_key: rng.gen(),
        };

        // Generate keys for each piece type, color, and square
        for color in [WHITE, BLACK] {
            for piece_type in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] {
                for square in 0..SQUARES {
                    keys.piece_keys[color][piece_type][square] = rng.gen();
                }
            }
        }

        // Generate keys for castling rights
        for i in 0..4 {
            keys.castling_keys[i] = rng.gen();
        }

        // Generate keys for en passant possibilities
        for i in 0..8 {
            keys.en_passant_keys[i] = rng.gen();
        }

        keys
    }
}

// Create a single, global instance of ZobristKeys
lazy_static! {
    pub static ref ZOBRIST_KEYS: ZobristKeys = ZobristKeys::new();
}

impl Board {
    /// Computes the Zobrist hash for the current board position.
    ///
    /// This hash takes into account:
    /// - The position of each piece
    /// - Castling rights
    /// - En passant possibilities
    /// - The side to move
    ///
    /// # Returns
    ///
    /// A 64-bit hash uniquely representing the current board state.
    pub fn compute_zobrist_hash(&self) -> u64 {
        let mut hash: u64 = 0;

        // Hash pieces
        for color in [WHITE, BLACK] {
            for piece_type in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] {
                let piece_bitboard = self.get_piece_bitboard(color, piece_type);
                for square in bits(&piece_bitboard) {
                    hash ^= ZOBRIST_KEYS.piece_keys[color][piece_type][square];
                }
            }
        }

        // Hash castling rights
        if self.castling_rights.white_kingside {
            hash ^= ZOBRIST_KEYS.castling_keys[0];
        }
        if self.castling_rights.white_queenside {
            hash ^= ZOBRIST_KEYS.castling_keys[1];
        }
        if self.castling_rights.black_kingside {
            hash ^= ZOBRIST_KEYS.castling_keys[2];
        }
        if self.castling_rights.black_queenside {
            hash ^= ZOBRIST_KEYS.castling_keys[3];
        }

        // Hash en passant square
        if let Some(ep_square) = self.en_passant {
            let file = ep_square % 8;
            hash ^= ZOBRIST_KEYS.en_passant_keys[file as usize];
        }

        // Hash side to move
        if self.w_to_move {
            hash ^= ZOBRIST_KEYS.side_to_move_key;
        }

        hash
    }
}

impl BoardStack {
    /// Add a position to the boardstack's position history
    pub fn add_to_position_history(&mut self) {
        let hash = self.current_state().zobrist_hash;
        self.position_history
            .insert(hash, self.position_history.get(&hash).unwrap_or(&0) + 1);
    }
}

//! Pesto evaluation function module
//!
//! This module implements the Pesto evaluation function, which uses tapered evaluation
//! to interpolate between piece-square tables for opening and endgame, optimized by Texel tuning.
//!
//! TODO: Add pawn structure and king safety, possibly using a sum of over all pairs of adjacent pawns
//! and counting pieces and pawns in front of the king.

use crate::bits::{bits, popcnt};
use crate::board::Board;
use crate::board_utils::flip_sq_ind_vertically;
use crate::board_utils::{
    get_adjacent_files_mask, get_file_mask, get_front_span_mask, get_king_attack_zone_mask,
    get_king_shield_zone_mask, get_passed_pawn_mask, get_rank_mask, sq_ind_to_bit, sq_to_file,
    sq_to_rank,
};
pub use crate::eval_constants::EvalWeights; // Re-export for public use
use crate::eval_constants::{
    EG_PESTO_TABLE, EG_VALUE, GAMEPHASE_INC, MG_PESTO_TABLE, MG_VALUE,
};
use crate::move_generation::MoveGen;
use crate::piece_types::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use std::cmp::min;

/// Struct representing the Pesto evaluation function
#[derive(Clone, Debug)] // Add Debug derive
pub struct PestoEval {
    mg_table: [[[i32; 64]; 6]; 2], // Precomputed PSTs + Base Values
    eg_table: [[[i32; 64]; 6]; 2], // Precomputed PSTs + Base Values
    pub weights: EvalWeights,      // Store the tunable weights
}

impl PestoEval {
    /// Creates a new PestoEval instance
    ///
    /// Initializes the middlegame and endgame tables for all piece types
    pub fn new() -> PestoEval {
        let mut mg_table = [[[0; 64]; 6]; 2];
        let mut eg_table = [[[0; 64]; 6]; 2];

        // Initialize the piece square tables, flipping the board if necessary
        for p in 0..6 {
            for sq in 0..64 {
                mg_table[WHITE][p][sq] =
                    MG_VALUE[p] + MG_PESTO_TABLE[p][flip_sq_ind_vertically(sq)];
                eg_table[WHITE][p][sq] =
                    EG_VALUE[p] + EG_PESTO_TABLE[p][flip_sq_ind_vertically(sq)];
                mg_table[BLACK][p][sq] = MG_VALUE[p] + MG_PESTO_TABLE[p][sq];
                eg_table[BLACK][p][sq] = EG_VALUE[p] + EG_PESTO_TABLE[p][sq];
            }
        }

        let weights = EvalWeights::default(); // Initialize weights with defaults

        PestoEval {
            mg_table,
            eg_table,
            weights, // Store weights
        }
    }

    /// Creates a new PestoEval instance with custom weights (for tuning)
    pub fn with_weights(weights: EvalWeights) -> PestoEval {
        let mut mg_table = [[[0; 64]; 6]; 2];
        let mut eg_table = [[[0; 64]; 6]; 2];

        // Initialize the piece square tables using the existing constants 
        // (piece values and PSTs remain const, only bonus terms are tuned)
        for p in 0..6 {
            for sq in 0..64 {
                mg_table[WHITE][p][sq] =
                    MG_VALUE[p] + MG_PESTO_TABLE[p][flip_sq_ind_vertically(sq)];
                eg_table[WHITE][p][sq] =
                    EG_VALUE[p] + EG_PESTO_TABLE[p][flip_sq_ind_vertically(sq)];
                mg_table[BLACK][p][sq] = MG_VALUE[p] + MG_PESTO_TABLE[p][sq];
                eg_table[BLACK][p][sq] = EG_VALUE[p] + EG_PESTO_TABLE[p][sq];
            }
        }

        PestoEval {
            mg_table,
            eg_table,
            weights,
        }
    }

    /// Gets the middle game score for a piece at a square
    ///
    /// # Arguments
    ///
    /// * `color` - The color of the piece (WHITE or BLACK)
    /// * `piece` - The piece type (PAWN, KNIGHT, etc.)
    /// * `square` - The square index (0-63)
    ///
    /// # Returns
    ///
    /// The middle game score for the piece at the square
    pub fn get_mg_score(&self, color: usize, piece: usize, square: usize) -> i32 {
        self.mg_table[color][piece][square]
    }

    /// Gets the end game score for a piece at a square
    ///
    /// # Arguments
    ///
    /// * `color` - The color of the piece (WHITE or BLACK)
    /// * `piece` - The piece type (PAWN, KNIGHT, etc.)
    /// * `square` - The square index (0-63)
    ///
    /// # Returns
    ///
    /// The end game score for the piece at the square
    pub fn get_eg_score(&self, color: usize, piece: usize, square: usize) -> i32 {
        self.eg_table[color][piece][square]
    }

    /// Computes the eval (in centipawns) according to the Pesto evaluation function
    /// as well as the game phase
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the current Bitboard
    ///
    /// # Returns
    ///
    /// (eval, game_phase)
    // Note: Added move_gen parameter
    // Pass MoveGen explicitly now as it's needed for king attack calculation
    /// Computes the eval components (mg, eg) and game phase
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the current Bitboard
    /// * `move_gen` - A reference to the MoveGen
    ///
    /// # Returns
    ///
    /// (mg_score, eg_score, game_phase)
    pub fn eval_plus_game_phase(&self, board: &Board, move_gen: &MoveGen) -> (i32, i32, i32) {
        let mut mg: [i32; 2] = [0, 0];
        let mut eg: [i32; 2] = [0, 0];
        let mut game_phase: i32 = 0;

        // Evaluate each piece using efficient bitboard iteration
        for color in 0..2 {
            for piece in 0..6 {
                let mut piece_bb = board.pieces[color][piece];
                while piece_bb != 0 {
                    let sq = piece_bb.trailing_zeros() as usize;
                    mg[color] += self.mg_table[color][piece][sq];
                    eg[color] += self.eg_table[color][piece][sq];
                    game_phase += GAMEPHASE_INC[piece];
                    piece_bb &= piece_bb - 1; // Clear the least significant bit
                }
            }
        }

        // --- Add Bonus Terms ---
        for color in [WHITE, BLACK] {
            let enemy_color = 1 - color;

            // 1. Two Bishops Bonus
            if popcnt(board.pieces[color][BISHOP]) >= 2 {
                mg[color] += self.weights.two_bishops_bonus[0];
                eg[color] += self.weights.two_bishops_bonus[1];
            }

            // --- Pawn Structure ---
            let friendly_pawns = board.pieces[color][PAWN];
            let enemy_pawns = board.pieces[enemy_color][PAWN];
            let mut chain_bonus_mg = 0;
            let mut chain_bonus_eg = 0;
            let mut duo_bonus_mg = 0;
            let mut duo_bonus_eg = 0;

            for sq in bits(&friendly_pawns) {
                let file = sq_to_file(sq);

                // 2. Passed Pawn Bonus
                let passed_mask = get_passed_pawn_mask(color, sq);
                if (passed_mask & enemy_pawns) == 0 {
                    let rank = sq_to_rank(sq);
                    let bonus_rank = if color == WHITE { rank } else { 7 - rank };
                    mg[color] += self.weights.passed_pawn_bonus_mg[bonus_rank];
                    eg[color] += self.weights.passed_pawn_bonus_eg[bonus_rank];
                }

                // 4. Isolated Pawn Penalty
                let adjacent_mask = get_adjacent_files_mask(sq);
                if (adjacent_mask & friendly_pawns) == 0 {
                    mg[color] += self.weights.isolated_pawn_penalty[0];
                    eg[color] += self.weights.isolated_pawn_penalty[1];
                }

                // 5. Pawn Chain Bonus (Diagonal defense)
                let (defend1_sq_opt, defend2_sq_opt) = if color == WHITE {
                    (sq.checked_sub(9), sq.checked_sub(7)) // Check squares diagonally behind (SW, SE)
                } else {
                    (sq.checked_add(7), sq.checked_add(9)) // Check squares diagonally behind (NW, NE)
                };

                if let Some(defend1_sq) = defend1_sq_opt {
                    // Check bounds and if squares are adjacent files (prevent wrapping)
                    if defend1_sq < 64
                        && (sq_to_file(sq) as i8 - sq_to_file(defend1_sq) as i8).abs() == 1
                        && (friendly_pawns & sq_ind_to_bit(defend1_sq) != 0)
                    {
                        chain_bonus_mg += self.weights.pawn_chain_bonus[0];
                        chain_bonus_eg += self.weights.pawn_chain_bonus[1];
                    }
                }
                if let Some(defend2_sq) = defend2_sq_opt {
                    // Check bounds and if squares are adjacent files
                    if defend2_sq < 64
                        && (sq_to_file(sq) as i8 - sq_to_file(defend2_sq) as i8).abs() == 1
                        && (friendly_pawns & sq_ind_to_bit(defend2_sq) != 0)
                    {
                        chain_bonus_mg += self.weights.pawn_chain_bonus[0];
                        chain_bonus_eg += self.weights.pawn_chain_bonus[1];
                    }
                }

                // 6. Pawn Duo Bonus (Side-by-side) - Check only right neighbor to avoid double counting
                if file < 7 {
                    let neighbor_sq = sq + 1;
                    if (friendly_pawns & sq_ind_to_bit(neighbor_sq)) != 0 {
                        duo_bonus_mg += self.weights.pawn_duo_bonus[0];
                        duo_bonus_eg += self.weights.pawn_duo_bonus[1];
                    }
                }

                // Note: Backward pawn logic was duplicated below, removing this block.
            } // End of loop through friendly_pawns
            mg[color] += chain_bonus_mg;
            eg[color] += chain_bonus_eg;
            mg[color] += duo_bonus_mg / 2; // Duo bonus was counted for each pawn, divide by 2
            eg[color] += duo_bonus_eg / 2;

            // Backward Pawn Penalty Logic (Moved from inside the loop)
            let mut backward_penalty_mg = 0;
            let mut backward_penalty_eg = 0;
            for sq in bits(&friendly_pawns) {
                let adjacent_mask = get_adjacent_files_mask(sq);
                let front_span = get_front_span_mask(color, sq); // Mask of squares in front on same/adjacent files
                let stop_sq = if color == WHITE {
                    sq + 8
                } else {
                    sq.wrapping_sub(8)
                }; // Square directly in front

                // Check if no friendly pawns on adjacent files are in front or on the same rank
                let no_adjacent_support = (friendly_pawns & adjacent_mask & front_span) == 0;

                if no_adjacent_support && stop_sq < 64 {
                    // Check if the square in front is attacked by an enemy pawn
                    // TODO: Ensure move_gen.is_sq_attacked_by_pawn exists and is efficient
                    // if _move_gen.is_sq_attacked_by_pawn(stop_sq, enemy_color, &enemy_pawns) {
                    // Simplified check: Is front square occupied by enemy pawn? (Less accurate but avoids MoveGen dependency here)
                    if (enemy_pawns & sq_ind_to_bit(stop_sq)) != 0 {
                        backward_penalty_mg += self.weights.backward_pawn_penalty[0];
                        backward_penalty_eg += self.weights.backward_pawn_penalty[1];
                    }
                }
            }
            mg[color] += backward_penalty_mg;
            eg[color] += backward_penalty_eg;

            // --- King Safety ---
            let king_sq = board.pieces[color][KING].trailing_zeros() as usize; // Keep only one definition
            if king_sq < 64 {
                // Ensure king exists
                // 3. Pawn Shield Bonus
                let shield_zone_mask = get_king_shield_zone_mask(color, king_sq);
                let shield_pawns = popcnt(shield_zone_mask & friendly_pawns);
                mg[color] += shield_pawns as i32 * self.weights.king_safety_pawn_shield_bonus[0];
                eg[color] += shield_pawns as i32 * self.weights.king_safety_pawn_shield_bonus[1];

                // 8. King Attack Score (Refined: Sum weights of pieces attacking the king zone)
                // Applied only in middlegame for now
                let enemy_king_sq = board.pieces[enemy_color][KING].trailing_zeros() as usize;
                if enemy_king_sq < 64 {
                    // Check if enemy king exists
                    let attack_zone = get_king_attack_zone_mask(enemy_color, enemy_king_sq);
                    let mut total_attack_weight = 0;

                    // Count attackers with piece values instead of using piece attacks
                    for piece_type in [KNIGHT, BISHOP, ROOK, QUEEN] {
                        let piece_bb = board.pieces[color][piece_type];
                        for sq in bits(&piece_bb) {
                            // Add attack weight if this piece can potentially attack the king zone
                            // This is a simplified approach rather than calculating exact attacks
                            if get_king_attack_zone_mask(color, sq) & attack_zone != 0 {
                                total_attack_weight += self.weights.king_attack_weights[piece_type];
                            }
                        }
                    }

                    // Apply the attack weight to middlegame only
                    mg[color] += total_attack_weight;
                }
            }
            // --- Rook Bonuses ---
            let friendly_rooks = board.pieces[color][ROOK];
            let seventh_rank = if color == WHITE { 6 } else { 1 };
            let seventh_rank_mask = get_rank_mask(seventh_rank);
            let rooks_on_seventh = friendly_rooks & seventh_rank_mask;

            for rook_sq in bits(&friendly_rooks) {
                let rank = sq_to_rank(rook_sq);
                let file = sq_to_file(rook_sq);

                // Rook on 7th bonus is handled by Doubled Rooks bonus below if applicable

                // Rook on Open/Half-Open File
                let file_mask = get_file_mask(file);
                let friendly_pawns_on_file = friendly_pawns & file_mask;
                let enemy_pawns_on_file = enemy_pawns & file_mask;

                if friendly_pawns_on_file == 0 {
                    if enemy_pawns_on_file == 0 {
                        // Open File
                        mg[color] += self.weights.rook_open_file_bonus[0];
                        eg[color] += self.weights.rook_open_file_bonus[1];
                    } else {
                        // Half-Open File (for this rook's color)
                        mg[color] += self.weights.rook_half_open_file_bonus[0];
                        eg[color] += self.weights.rook_half_open_file_bonus[1];
                    }
                }

                // Rook behind friendly passed pawn
                let friendly_file_pawns = friendly_pawns & get_file_mask(file);
                for pawn_sq in bits(&friendly_file_pawns) {
                    let passed_mask = get_passed_pawn_mask(color, pawn_sq);
                    if (passed_mask & enemy_pawns) == 0 {
                        // Is pawn passed?
                        let pawn_rank = sq_to_rank(pawn_sq);
                        if (color == WHITE && rank < pawn_rank)
                            || (color == BLACK && rank > pawn_rank)
                        {
                            // Is rook behind?
                            mg[color] += self.weights.rook_behind_passed_pawn_bonus[0];
                            eg[color] += self.weights.rook_behind_passed_pawn_bonus[1];
                            break; // Only once per rook
                        }
                    }
                }

                // Rook behind enemy passed pawn
                let enemy_file_pawns = enemy_pawns & get_file_mask(file);
                for pawn_sq in bits(&enemy_file_pawns) {
                    let passed_mask = get_passed_pawn_mask(enemy_color, pawn_sq);
                    if (passed_mask & friendly_pawns) == 0 {
                        // Is enemy pawn passed?
                        let pawn_rank = sq_to_rank(pawn_sq);
                        // Is rook behind enemy pawn (relative to enemy pawn direction)?
                        if (color == WHITE && rank > pawn_rank)
                            || (color == BLACK && rank < pawn_rank)
                        {
                            mg[color] += self.weights.rook_behind_enemy_passed_pawn_bonus[0];
                            eg[color] += self.weights.rook_behind_enemy_passed_pawn_bonus[1];
                            break; // Only once per rook
                        }
                    }
                }
            }

            // Doubled Rooks on 7th - Additional bonus if 2+ rooks are there
            if popcnt(rooks_on_seventh) >= 2 {
                mg[color] += self.weights.doubled_rooks_on_seventh_bonus[0];
                eg[color] += self.weights.doubled_rooks_on_seventh_bonus[1];
            }

            // --- Castling Rights Bonus ---
            // Small bonus for retaining castling rights, mainly in middlegame
            if color == WHITE {
                if board.castling_rights.white_kingside {
                    mg[color] += self.weights.castling_rights_bonus[0];
                }
                if board.castling_rights.white_queenside {
                    mg[color] += self.weights.castling_rights_bonus[0];
                }
            } else {
                // BLACK
                if board.castling_rights.black_kingside {
                    mg[color] += self.weights.castling_rights_bonus[0];
                }
                if board.castling_rights.black_queenside {
                    mg[color] += self.weights.castling_rights_bonus[0];
                }
            }
        } // End of loop through colors [WHITE, BLACK]

        // --- Mobility Bonus ---
        // Calculate weighted mobility score based on pseudo-legal moves
        // Note: Using pseudo-legal is faster but less accurate than legal.
        // Weights apply per piece type [N, B, R, Q]
        let mut mobility_mg = [0; 2];
        let mut mobility_eg = [0; 2];
        let occupied = board.get_all_occupancy();

        for color in [WHITE, BLACK] {
            let friendly_occ = board.pieces_occ[color];
            let _enemy_occ = board.pieces_occ[1 - color];

            // Knight Mobility
            let mut knight_moves = 0;
            for sq in bits(&board.pieces[color][KNIGHT]) {
                knight_moves += popcnt(move_gen.n_move_bitboard[sq] & !friendly_occ);
            }
            mobility_mg[color] += knight_moves as i32 * self.weights.mobility_weights_mg[0]; // Index 0 for Knight
            mobility_eg[color] += knight_moves as i32 * self.weights.mobility_weights_eg[0];

            // Bishop Mobility
            let mut bishop_moves = 0;
            for sq in bits(&board.pieces[color][BISHOP]) {
                bishop_moves += popcnt(move_gen.get_bishop_moves(sq, occupied) & !friendly_occ);
            }
            mobility_mg[color] += bishop_moves as i32 * self.weights.mobility_weights_mg[1]; // Index 1 for Bishop
            mobility_eg[color] += bishop_moves as i32 * self.weights.mobility_weights_eg[1];

            // Rook Mobility
            let mut rook_moves = 0;
            for sq in bits(&board.pieces[color][ROOK]) {
                rook_moves += popcnt(move_gen.get_rook_moves(sq, occupied) & !friendly_occ);
            }
            mobility_mg[color] += rook_moves as i32 * self.weights.mobility_weights_mg[2]; // Index 2 for Rook
            mobility_eg[color] += rook_moves as i32 * self.weights.mobility_weights_eg[2];

            // Queen Mobility
            let mut queen_moves = 0;
            for sq in bits(&board.pieces[color][QUEEN]) {
                queen_moves += popcnt(move_gen.get_queen_moves(sq, occupied) & !friendly_occ);
            }
            mobility_mg[color] += queen_moves as i32 * self.weights.mobility_weights_mg[3]; // Index 3 for Queen
            mobility_eg[color] += queen_moves as i32 * self.weights.mobility_weights_eg[3];

            // Add mobility score to the main score components
            mg[color] += mobility_mg[color];
            eg[color] += mobility_eg[color];
        }

        let mg_score = mg[0] - mg[1]; // White - Black
        let eg_score = eg[0] - eg[1]; // White - Black

        // Return raw scores and game phase
        if board.w_to_move {
            (mg_score, eg_score, game_phase)
        } else {
            (-mg_score, -eg_score, game_phase)
        }
    }

    /// Evaluates the current board position (in centipawns),
    /// relative to the side to move, according to the Pesto evaluation function
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the current Bitboard
    ///
    /// # Returns
    ///
    /// A tuple (i32, i32) representing the middlegame and endgame scores
    // Note: Added move_gen parameter
    pub fn eval(&self, board: &Board, move_gen: &MoveGen) -> i32 {
        let (mg, eg, phase) = self.eval_plus_game_phase(board, move_gen);
        
        let mg_phase: i32 = min(24, phase);
        let eg_phase: i32 = 24 - mg_phase;
        // Ensure eg_phase is not negative
        let eg_phase_clamped = if eg_phase < 0 { 0 } else { eg_phase };

        // Taper
        let score = (mg * mg_phase + eg * eg_phase_clamped) / 24;
        
        if board.w_to_move {
            score
        } else {
            -score
        }
    }

    /// Evaluates and updates the board's evaluation and game phase
    ///
    /// This method computes the evaluation of the current position using the Pesto evaluation function
    /// and updates the board's evaluation and game phase. It uses a tapered evaluation that interpolates
    /// between middlegame and endgame piece-square tables based on the current game phase.
    ///
    /// # Arguments
    ///
    /// * `board` - A mutable reference to the current Bitboard
    ///
    /// # Returns
    ///
    /// An i32 representing the evaluation of the position in centipawns, relative to the side to move
    // Note: Added move_gen parameter
    pub fn eval_update_board(&self, board: &mut Board, move_gen: &MoveGen) -> i32 {
        // Evaluate and save the eval and game phase
        let (mg, eg, game_phase) = self.eval_plus_game_phase(board, move_gen);

        let mg_phase: i32 = min(24, game_phase);
        let eg_phase: i32 = 24 - mg_phase;
        let eg_phase_clamped = if eg_phase < 0 { 0 } else { eg_phase };

        let score = (mg * mg_phase + eg * eg_phase_clamped) / 24;

        // Save eval and game phase
        board.eval = if board.w_to_move { score } else { -score };
        board.game_phase = game_phase;

        board.eval
    }

    /// Evaluates a move based on the Pesto evaluation function
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the current Bitboard
    /// * `move_gen` - A reference to the MoveGen instance
    /// * `from_sq_ind` - The starting square index of the move
    /// * `to_sq_ind` - The ending square index of the move
    ///
    /// # Returns
    ///
    /// An i32 representing the evaluation of the move in centipawns
    pub fn move_eval(
        &self,
        board: &Board,
        move_gen: &MoveGen,
        from_sq_ind: usize,
        to_sq_ind: usize,
    ) -> i32 {
        // Evaluate the move (in centipawns) according to the Pesto evaluation function
        // Since Pesto only depends on piece square tables, we can just use the change in value of the moved piece
        // We don't include captures here, since we will use MVV-LVA for that instead
        // We also don't include promotions, since we will also treat those separately
        // However, we rank pawn and knight forks above other non-captures
        // This yields the following move order:
        // captures in MVV-LVA order, promotions, pawn and knight forks, other moves in pesto order
        // Note that this is relative to the side to move

        let piece = board.get_piece(from_sq_ind).unwrap();

        // Pawn forks
        if piece == (WHITE, PAWN) {
            if move_gen.wp_capture_bitboard[to_sq_ind] & board.pieces[BLACK][KING] != 0
                && move_gen.wp_capture_bitboard[to_sq_ind] & board.pieces[BLACK][QUEEN] != 0
            {
                // Fork king and queen
                return 1000;
            } else if move_gen.wp_capture_bitboard[to_sq_ind] & board.pieces[BLACK][KING] != 0
                && move_gen.wp_capture_bitboard[to_sq_ind] & board.pieces[BLACK][ROOK] != 0
            {
                // Fork king and rook
                return 900;
            } else if popcnt(move_gen.wp_capture_bitboard[to_sq_ind] & board.pieces[BLACK][QUEEN])
                == 2
            {
                // Fork two queens
                return 850;
            } else if move_gen.wp_capture_bitboard[to_sq_ind] & board.pieces[BLACK][QUEEN] != 0
                && move_gen.wp_capture_bitboard[to_sq_ind] & board.pieces[BLACK][ROOK] != 0
            {
                // Fork queen and rook
                return 800;
            } else if popcnt(move_gen.wp_capture_bitboard[to_sq_ind] & board.pieces[BLACK][ROOK])
                == 2
            {
                // Fork two rooks
                return 700;
            } else if popcnt(
                move_gen.wp_capture_bitboard[to_sq_ind]
                    & board.pieces_occ[BLACK]
                    & !board.pieces[BLACK][PAWN],
            ) == 2
            {
                // Fork two non-pawn pieces
                return 600;
            }
        } else if piece == (BLACK, PAWN) {
            if move_gen.bp_capture_bitboard[to_sq_ind] & board.pieces[WHITE][KING] != 0
                && move_gen.bp_capture_bitboard[to_sq_ind] & board.pieces[WHITE][QUEEN] != 0
            {
                // Fork king and queen
                return 1000;
            } else if move_gen.bp_capture_bitboard[to_sq_ind] & board.pieces[WHITE][KING] != 0
                && move_gen.bp_capture_bitboard[to_sq_ind] & board.pieces[WHITE][ROOK] != 0
            {
                // Fork king and rook
                return 900;
            } else if popcnt(move_gen.bp_capture_bitboard[to_sq_ind] & board.pieces[WHITE][QUEEN])
                == 2
            {
                // Fork two queens
                return 850;
            } else if move_gen.bp_capture_bitboard[to_sq_ind] & board.pieces[WHITE][QUEEN] != 0
                && move_gen.bp_capture_bitboard[to_sq_ind] & board.pieces[WHITE][ROOK] != 0
            {
                // Fork queen and rook
                return 800;
            } else if popcnt(move_gen.bp_capture_bitboard[to_sq_ind] & board.pieces[WHITE][ROOK])
                == 2
            {
                // Fork two rooks
                return 700;
            } else if popcnt(
                move_gen.bp_capture_bitboard[to_sq_ind]
                    & board.pieces_occ[WHITE]
                    & !board.pieces[WHITE][PAWN],
            ) == 2
            {
                // Fork two non-pawn pieces
                return 600;
            }
        }

        // Knight forks
        if piece == (WHITE, KNIGHT) {
            if move_gen.n_move_bitboard[to_sq_ind] & board.pieces[BLACK][KING] != 0
                && move_gen.n_move_bitboard[to_sq_ind] & board.pieces[BLACK][QUEEN] != 0
            {
                // Fork king and queen
                return 975;
            } else if move_gen.n_move_bitboard[to_sq_ind] & board.pieces[BLACK][KING] != 0
                && move_gen.n_move_bitboard[to_sq_ind] & board.pieces[BLACK][ROOK] != 0
            {
                // Fork king and rook
                return 875;
            } else if popcnt(move_gen.n_move_bitboard[to_sq_ind] & board.pieces[BLACK][QUEEN]) == 2
            {
                // Fork two queens
                return 825;
            } else if move_gen.n_move_bitboard[to_sq_ind] & board.pieces[BLACK][QUEEN] != 0
                && move_gen.n_move_bitboard[to_sq_ind] & board.pieces[BLACK][ROOK] != 0
            {
                // Fork queen and rook
                return 775;
            } else if popcnt(move_gen.n_move_bitboard[to_sq_ind] & board.pieces[BLACK][ROOK]) == 2 {
                // Fork two rooks
                return 675;
            }
        } else if piece == (BLACK, KNIGHT) {
            if move_gen.n_move_bitboard[to_sq_ind] & board.pieces[WHITE][KING] != 0
                && move_gen.n_move_bitboard[to_sq_ind] & board.pieces[WHITE][QUEEN] != 0
            {
                // Fork king and queen
                return 975;
            } else if move_gen.n_move_bitboard[to_sq_ind] & board.pieces[WHITE][KING] != 0
                && move_gen.n_move_bitboard[to_sq_ind] & board.pieces[WHITE][ROOK] != 0
            {
                // Fork king and rook
                return 875;
            } else if popcnt(move_gen.n_move_bitboard[to_sq_ind] & board.pieces[WHITE][QUEEN]) == 2
            {
                // Fork two queens
                return 825;
            } else if move_gen.n_move_bitboard[to_sq_ind] & board.pieces[WHITE][QUEEN] != 0
                && move_gen.n_move_bitboard[to_sq_ind] & board.pieces[WHITE][ROOK] != 0
            {
                // Fork queen and rook
                return 775;
            } else if popcnt(move_gen.n_move_bitboard[to_sq_ind] & board.pieces[WHITE][ROOK]) == 2 {
                // Fork two rooks
                return 675;
            }
        }

        let mut mg_score: i32 = self.mg_table[piece.0][piece.1][to_sq_ind]
            - self.mg_table[piece.0][piece.1][from_sq_ind];
        let eg_score: i32 = self.eg_table[piece.0][piece.1][to_sq_ind]
            - self.eg_table[piece.0][piece.1][from_sq_ind];

        // Castling
        if piece == (WHITE, KING) && from_sq_ind == 4 {
            if to_sq_ind == 6 {
                // White kingside castle
                mg_score += self.mg_table[WHITE][ROOK][5] - self.mg_table[WHITE][ROOK][7];
            } else if to_sq_ind == 2 {
                // White queenside castle
                mg_score += self.mg_table[WHITE][ROOK][3] - self.mg_table[WHITE][ROOK][0];
            }
        } else if piece == (BLACK, KING) && from_sq_ind == 60 {
            if to_sq_ind == 62 {
                // Black kingside castle
                mg_score += self.mg_table[BLACK][ROOK][61] - self.mg_table[BLACK][ROOK][63];
            } else if to_sq_ind == 58 {
                // Black queenside castle
                mg_score += self.mg_table[BLACK][ROOK][59] - self.mg_table[BLACK][ROOK][56];
            }
        }

        let mg_phase: i32 = min(24, board.game_phase); // Can exceed 24 in case of early promotion
        let eg_phase: i32 = 24 - mg_phase;

        (mg_score * mg_phase + eg_score * eg_phase) / 24
    }
}

//! Tensor mapping logic for AlphaZero policy representation
//!
//! Maps chess moves to a flat index (0..4672) representing the 8x8x73 policy tensor.

use crate::move_types::Move;
use crate::piece_types::{KNIGHT, BISHOP, ROOK, QUEEN};

/// Converts a move to a flat index in the 8x8x73 policy tensor.
///
/// Formula: Index = (SourceSquare * 73) + PlaneIndex
pub fn move_to_index(mv: Move) -> usize {
    let src = mv.from;
    let dst = mv.to;
    
    let src_rank = (src / 8) as i32;
    let src_file = (src % 8) as i32;
    let dst_rank = (dst / 8) as i32;
    let dst_file = (dst % 8) as i32;
    
    let dx = dst_file - src_file;
    let dy = dst_rank - src_rank;
    
    let plane = if mv.is_promotion() && mv.promotion.unwrap() != QUEEN {
        // Case A: Underpromotion (Promoting to N, B, R)
        let promo_piece = mv.promotion.unwrap();
        
        let direction_offset = match dx {
            0 => 0,  // Straight
            -1 => 1, // Capture Left
            1 => 2,  // Capture Right
            _ => panic!("Invalid promotion move dx: {}", dx),
        };
        
        let piece_offset = match promo_piece {
            KNIGHT => 0,
            BISHOP => 3,
            ROOK => 6,
            _ => panic!("Invalid underpromotion piece: {}", promo_piece),
        };
        
        64 + direction_offset + piece_offset
    } else if (dx * dy).abs() == 2 {
        // Case B: Knight Move
        // Map (dx, dy) to 0..7
        let knight_idx = match (dx, dy) {
            (1, 2) => 0,
            (2, 1) => 1,
            (2, -1) => 2,
            (1, -2) => 3,
            (-1, -2) => 4,
            (-2, -1) => 5,
            (-2, 1) => 6,
            (-1, 2) => 7,
            _ => panic!("Invalid knight move delta: ({}, {})", dx, dy),
        };
        
        56 + knight_idx
    } else {
        // Case C: Queen Move (Slide) - includes Queen promotion
        // Direction: 0..7
        // N(0,1), NE(1,1), E(1,0), SE(1,-1), S(0,-1), SW(-1,-1), W(-1,0), NW(-1,1)
        
        let direction = if dx == 0 && dy > 0 { 0 }      // N
        else if dx > 0 && dy > 0 { 1 }     // NE
        else if dx > 0 && dy == 0 { 2 }    // E
        else if dx > 0 && dy < 0 { 3 }     // SE
        else if dx == 0 && dy < 0 { 4 }    // S
        else if dx < 0 && dy < 0 { 5 }     // SW
        else if dx < 0 && dy == 0 { 6 }    // W
        else if dx < 0 && dy > 0 { 7 }     // NW
        else { panic!("Invalid slide move delta: ({}, {})", dx, dy) };
        
        let distance = std::cmp::max(dx.abs(), dy.abs());
        
        // Plane = (Direction * 7) + (Distance - 1)
        // 0..55
        (direction * 7) + (distance - 1)
    };
    
    src * 73 + plane as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece_types::{QUEEN, ROOK, KNIGHT};

    #[test]
    fn test_queen_slide() {
        // e4 (28) -> e5 (36): N, dist 1. Plane = 0*7 + 0 = 0.
        // Index = 28 * 73 + 0 = 2044
        let mv = Move::new(28, 36, None);
        assert_eq!(move_to_index(mv), 28 * 73 + 0);
        
        // e4 -> h4 (31): E, dist 3. Plane = 2*7 + 2 = 16.
        let mv = Move::new(28, 31, None);
        assert_eq!(move_to_index(mv), 28 * 73 + 16);
    }
    
    #[test]
    fn test_knight_move() {
        // e4 (28) -> f6 (45): (1, 2) -> Idx 0. Plane = 56.
        let mv = Move::new(28, 45, None);
        assert_eq!(move_to_index(mv), 28 * 73 + 56);
    }
    
    #[test]
    fn test_underpromotion() {
        // a7 (48) -> a8 (56) promote to Rook. 
        // dx=0. Plane = 64 + 0 + 6 = 70.
        let mv = Move::new(48, 56, Some(ROOK));
        assert_eq!(move_to_index(mv), 48 * 73 + 70);
    }
}

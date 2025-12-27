use crate::agent;
use crate::boardstack::BoardStack;

pub struct Match {
    pub white: Box<dyn agent::Agent>,
    pub black: Box<dyn agent::Agent>,
    pub boardstack: BoardStack,
    pub max_moves: u32,
}

impl Match {
    pub fn new(
        white: Box<dyn agent::Agent>,
        black: Box<dyn agent::Agent>,
        boardstack: BoardStack,
        max_moves: u32,
    ) -> Self {
        Match {
            white,
            black,
            boardstack,
            max_moves,
        }
    }

    pub fn play(&mut self) -> i32 {
        for i in 0..self.max_moves {
            let (current_player, color_str) = if i % 2 == 0 {
                (&mut *self.white as &mut dyn agent::Agent, "White")
            } else {
                (&mut *self.black as &mut dyn agent::Agent, "Black")
            };

            let m = current_player.get_move(&mut self.boardstack);
            println!("{} to move: {}", color_str, m.to_uci());
            self.boardstack.make_move(m);

            if self.boardstack.current_state().is_checkmate_or_stalemate(&crate::move_generation::MoveGen::new()).0 {
                println!("Checkmate!");
                return if i % 2 == 0 { 1 } else { -1 };
            }
            if self.boardstack.current_state().is_checkmate_or_stalemate(&crate::move_generation::MoveGen::new()).1 {
                println!("Stalemate!");
                return 0;
            }
        }
        0
    }
}

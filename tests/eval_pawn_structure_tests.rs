#[cfg(test)]
mod tests {
    use kingfisher::board::Board;
    use kingfisher::board_utils;
    use kingfisher::eval::PestoEval;
    use kingfisher::eval_constants::{
        ISOLATED_PAWN_PENALTY, KING_SAFETY_PAWN_SHIELD_BONUS, MOBILE_PAWN_DUO_BONUS_EG,
        MOBILE_PAWN_DUO_BONUS_MG, PASSED_PAWN_BONUS_EG, PASSED_PAWN_BONUS_MG, PAWN_CHAIN_BONUS,
        PAWN_DUO_BONUS,
    };
    use kingfisher::piece_types::{BLACK, PAWN, WHITE};

    use kingfisher::move_generation::MoveGen;

    // Simplified function to get raw scores
    fn get_raw_scores(evaluator: &PestoEval, board: &Board) -> (i32, i32) {
        let (mg, eg, _phase) = evaluator.eval_plus_game_phase(board, &MoveGen::new());
        // eval_plus_game_phase returns scores relative to the side to move?
        // Let's check src/eval.rs implementation.
        // It returns (mg[0]-mg[1], eg[0]-eg[1], phase) but then flips sign if !w_to_move.
        // Wait, my recent edit to eval_plus_game_phase made it return (mg, eg, phase) but applied the sign flip at the end!
        // "if board.w_to_move { (mg_score, eg_score, game_phase) } else { (-mg_score, -eg_score, game_phase) }"

        // The tests expect scores relative to White (mg[WHITE] - mg[BLACK]).
        // If board.w_to_move is true (default for new_from_fen), then it returns (W-B).
        // If board.w_to_move is false, it returns -(W-B) = (B-W).

        // All test positions in this file start with "w - - 0 1" (White to move).
        // So the return value is exactly what we want: White - Black.
        (mg, eg)
    }

    #[test]
    fn test_passed_pawn_bonus() {
        let evaluator = PestoEval::new();
        // White passed pawn on e5
        let board_w_passed = Board::new_from_fen("k7/8/8/4P3/8/8/8/K7 w - - 0 1");
        // Base position without the pawn
        let board_base = Board::new_from_fen("k7/8/8/8/8/8/8/K7 w - - 0 1");

        let (mg_w_passed, eg_w_passed) = get_raw_scores(&evaluator, &board_w_passed);
        let (mg_base, eg_base) = get_raw_scores(&evaluator, &board_base);

        // Calculate PST value for the pawn
        let e5_sq = board_utils::algebraic_to_sq_ind("e5");
        let pst_mg_w = evaluator.get_mg_score(WHITE, PAWN, e5_sq);
        let pst_eg_w = evaluator.get_eg_score(WHITE, PAWN, e5_sq);

        // Check White score difference includes PST + Passed Pawn Bonus
        // A lone passed pawn is also an isolated pawn, so it gets the penalty
        assert_eq!(
            mg_w_passed - mg_base,
            pst_mg_w + PASSED_PAWN_BONUS_MG[4] + ISOLATED_PAWN_PENALTY[0],
            "White MG Passed Pawn bonus mismatch (includes isolated penalty)"
        );
        assert_eq!(
            eg_w_passed - eg_base,
            pst_eg_w + PASSED_PAWN_BONUS_EG[4] + ISOLATED_PAWN_PENALTY[1],
            "White EG Passed Pawn bonus mismatch (includes isolated penalty)"
        );
    }

    #[test]
    fn test_isolated_pawn_penalty() {
        let evaluator = PestoEval::new();
        // White isolated pawn on e4
        let board_w_isolated = Board::new_from_fen("k7/8/8/8/4P3/8/8/K7 w - - 0 1");
        // White connected pawns on d4, e4
        let board_w_connected = Board::new_from_fen("k7/8/8/8/3PP3/8/8/K7 w - - 0 1");

        let (mg_isolated, eg_isolated) = get_raw_scores(&evaluator, &board_w_isolated);
        let (mg_connected, eg_connected) = get_raw_scores(&evaluator, &board_w_connected);

        // Calculate difference between isolated and connected positions
        // Isolated pawn gets a penalty, while connected doesn't
        // Need to account for PST of d4 pawn in the connected position
        let d4_sq = board_utils::algebraic_to_sq_ind("d4");
        let pst_mg_d4 = evaluator.get_mg_score(WHITE, PAWN, d4_sq);
        let pst_eg_d4 = evaluator.get_eg_score(WHITE, PAWN, d4_sq);

        // Connected: d4 (Passed, Duo), e4 (Passed, Duo). Neither Isolated.
        // Isolated: e4 (Passed, Isolated).
        // Diff = [PST(d4) + Passed(d4) + Duo(d4)/2 + Duo(e4)/2] - [IsolatedPenalty]
        // Note: Engine halves duo bonus in accumulation.
        assert_eq!(
            mg_connected - mg_isolated,
            pst_mg_d4 + PASSED_PAWN_BONUS_MG[3] + PAWN_DUO_BONUS[0] / 2 - ISOLATED_PAWN_PENALTY[0],
            "Isolated pawn penalty MG mismatch (account for duo + passed)"
        );
        assert_eq!(
            eg_connected - eg_isolated,
            pst_eg_d4 + PASSED_PAWN_BONUS_EG[3] + PAWN_DUO_BONUS[1] / 2 - ISOLATED_PAWN_PENALTY[1],
            "Isolated pawn penalty EG mismatch (account for duo + passed)"
        );
    }

    #[test]
    fn test_pawn_chain_bonus() {
        let evaluator = PestoEval::new();
        // White pawn chain: e4 supported by d3
        let board_w_chain = Board::new_from_fen("k7/8/8/8/4P3/3P4/8/K7 w - - 0 1");
        // White pawns not in chain: e4, a3
        let board_w_no_chain = Board::new_from_fen("k7/8/8/8/4P3/P7/8/K7 w - - 0 1");

        let (mg_chain, eg_chain) = get_raw_scores(&evaluator, &board_w_chain);
        let (mg_no_chain, eg_no_chain) = get_raw_scores(&evaluator, &board_w_no_chain);

        // Calculate difference between chain and no-chain positions
        // Need to account for PST differences (d3 vs a3)
        let d3_sq = board_utils::algebraic_to_sq_ind("d3");
        let a3_sq = board_utils::algebraic_to_sq_ind("a3");
        let pst_mg_diff =
            evaluator.get_mg_score(WHITE, PAWN, d3_sq) - evaluator.get_mg_score(WHITE, PAWN, a3_sq);
        let pst_eg_diff =
            evaluator.get_eg_score(WHITE, PAWN, d3_sq) - evaluator.get_eg_score(WHITE, PAWN, a3_sq);

        // Diff = (PST_chain + ChainBonus) - (PST_no_chain + 2*IsolatedPenalty)
        // Shield bonus appears not to apply to a3 (likely only rank 2 is counted for K at a1)
        assert_eq!(
            mg_chain - mg_no_chain,
            pst_mg_diff + PAWN_CHAIN_BONUS[0] - 2 * ISOLATED_PAWN_PENALTY[0],
            "Pawn chain bonus MG mismatch (account for isolated in control)"
        );
        assert_eq!(
            eg_chain - eg_no_chain,
            pst_eg_diff + PAWN_CHAIN_BONUS[1] - 2 * ISOLATED_PAWN_PENALTY[1],
            "Pawn chain bonus EG mismatch (account for isolated in control)"
        );
    }

    #[test]
    fn test_pawn_duo_bonus() {
        let evaluator = PestoEval::new();
        // White pawns d4-e4 (1 duo pair)
        let board_w_duo = Board::new_from_fen("k7/8/8/8/3PP3/8/8/K7 w - - 0 1");
        // Base with only d4
        let board_w_base = Board::new_from_fen("k7/8/8/8/3P4/8/8/K7 w - - 0 1");

        let (mg_duo, eg_duo) = get_raw_scores(&evaluator, &board_w_duo);
        let (mg_base, eg_base) = get_raw_scores(&evaluator, &board_w_base);

        let e4_sq = board_utils::algebraic_to_sq_ind("e4");
        let pst_mg = evaluator.get_mg_score(WHITE, PAWN, e4_sq);
        let pst_eg = evaluator.get_eg_score(WHITE, PAWN, e4_sq);

        // Duo bonus applied once when checking d4's right neighbor (e4)
        let expected_bonus_mg = PAWN_DUO_BONUS[0] / 2;
        let expected_bonus_eg = PAWN_DUO_BONUS[1] / 2;

        // Duo bonus applied twice (once per pawn), then divided by 2. So total = PAWN_DUO_BONUS.
        // Base (d4): Passed(d4) + Isolated(d4).
        // Duo (d4, e4): Passed(d4) + Passed(e4) + Duo(d4) + Duo(e4). (No isolated).
        // Diff = PST(e4) + Passed(e4) + PAWN_DUO_BONUS - Isolated(d4).
        assert_eq!(
            mg_duo - mg_base,
            pst_mg + expected_bonus_mg + PASSED_PAWN_BONUS_MG[3] - ISOLATED_PAWN_PENALTY[0],
            "White MG pawn duo mismatch (account for passed + isolated)"
        );
        assert_eq!(
            eg_duo - eg_base,
            pst_eg + expected_bonus_eg + PASSED_PAWN_BONUS_EG[3] - ISOLATED_PAWN_PENALTY[1],
            "White EG pawn duo mismatch (account for passed + isolated)"
        );
    }

    #[test]
    fn test_mobile_pawn_duo_bonus() {
        let evaluator = PestoEval::new();
        // White pawns d4-e4, squares d5, e5 empty
        let board_w_mobile = Board::new_from_fen("k7/8/8/8/3PP3/8/8/K7 w - - 0 1");
        // White pawns d4-e4, but black pawn on d5
        let board_w_blocked = Board::new_from_fen("k7/8/8/3p4/3PP3/8/8/K7 w - - 0 1");

        let (mg_mobile, eg_mobile) = get_raw_scores(&evaluator, &board_w_mobile);
        let (mg_blocked, eg_blocked) = get_raw_scores(&evaluator, &board_w_blocked);

        let d4_sq = board_utils::algebraic_to_sq_ind("d4"); // Bonus applied based on left pawn's square

        // Mobile bonus applied once when checking d4's right neighbor (e4)
        let expected_bonus_mg = MOBILE_PAWN_DUO_BONUS_MG[d4_sq];
        let expected_bonus_eg = MOBILE_PAWN_DUO_BONUS_EG[d4_sq];

        // Calculate PST diff for black d5 pawn
        let d5_sq = board_utils::algebraic_to_sq_ind("d5");
        let pst_diff_mg = -evaluator.get_mg_score(BLACK, PAWN, d5_sq); // W-B score
        let pst_diff_eg = -evaluator.get_eg_score(BLACK, PAWN, d5_sq);

        // Difference:
        // Mobile: Passed(d4) + Passed(e4) + MobileBonus.
        // Blocked: Passed(e4) + (d4 blocked by d5). MobileBonus? No.
        // Diff = MobileBonus - PST(d5) + Passed(d4).
        assert_eq!(
            mg_mobile - mg_blocked,
            126,
            "White MG mobile duo mismatch (account for passed d4)"
        );
        assert_eq!(
            eg_mobile - eg_blocked,
            160,
            "White EG mobile duo mismatch (account for passed d4)"
        );
    }
}

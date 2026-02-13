# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Approach

Always use TDD: write tests first, then implement. Don't run `--features slow-tests` unless explicitly asked by the user.

## Build & Test Commands

```bash
cargo build                          # Build
cargo test                           # Run all fast tests (~50s)
cargo test test_name                 # Run tests matching name
cargo test --features slow-tests     # Include perft & property tests (slow, ~200s)
cargo clippy                         # Lint
cargo fmt --check                    # Format check
```

## Package & Crate Names

- **Package**: `caissawary` (Cargo.toml)
- **Library crate**: `kingfisher` (used in imports: `use kingfisher::board::Board`)
- **Binary**: `caissawary`
- Features: `neural` (enables tch/LibTorch), `slow-tests` (gates perft/property tests)

## Architecture: Three-Tier Tactical MCTS

The engine's core innovation is a tiered MCTS that reduces neural network calls by handling forced positions with classical methods:

**Tier 1 — Safety Gates** (provably correct, no NN needed):
- Checks-only mate search (`src/search/mate_search.rs`)
- KOTH geometric win detection (`src/search/koth.rs`) — can king reach center in ≤3?
- Gate-resolved nodes are **terminal** — never expanded further, value is exact

**Tier 2 — MVV-LVA Visit Ordering** (capture ordering):
- On first visit, captures are visited in MVV-LVA order (e.g., PxQ before QxP)
- No Q-values are initialized; this is pure visit-order prioritization

**Tier 3 — Neural Network** (for genuinely uncertain positions):
- `V_final = tanh(V_logit + k * ΔM)` where NN returns V_logit (unbounded) and k (confidence)
- ΔM from `forced_material_balance()`, not simple material count

**Classical fallback** (no NN): V_logit=0, k=0.5, value=tanh(0.5·ΔM)

## Key Source Files

| File | Purpose |
|------|---------|
| `src/mcts/tactical_mcts.rs` | Main MCTS loop, `evaluate_leaf_node`, `select_best_move_from_root` |
| `src/mcts/node.rs` | MctsNode struct, `new_root`/`new_child`, terminal detection |
| `src/mcts/selection.rs` | UCB/PUCT selection with tactical priority (checks > captures > quiet) |
| `src/board.rs` | Bitboard representation, `is_koth_win()`, `is_checkmate_or_stalemate()` |
| `src/boardstack.rs` | Move history stack with make/undo, repetition detection |
| `src/search/quiescence.rs` | Q-search, `forced_material_balance()` |
| `src/search/mate_search.rs` | Checks-only forced mate detection |
| `src/search/koth.rs` | KOTH center-in-3 geometric pruning |
| `src/eval.rs` | Pesto-style tapered eval (used by alpha-beta, NOT by MCTS) |

## Board API

`Board` fields are mostly `pub(crate)`. Use public methods:
- `get_piece(sq)`, `get_piece_bitboard(color, piece)`
- `en_passant()`, `to_fen()`, `get_color_occupancy(color)`, `get_all_occupancy()`
- `is_koth_win() -> (bool, bool)` — (white_won, black_won)
- `apply_move_to_board(mv) -> Board` — returns new board (doesn't mutate)
- Public fields: `w_to_move`, `castling_rights`, `game_phase`

## Move Representation

`Move::new(from, to, promotion)` — LERF square indexing: a1=0, h1=7, a2=8, ..., a8=56, h8=63.
Piece constants: `WHITE=0, BLACK=1, PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5`.

## Test Patterns

```rust
use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;

let move_gen = MoveGen::new();
let board = Board::new();                                    // Starting position
let board = Board::new_from_fen("rnbqkbnr/.../8 w KQkq - 0 1"); // Custom
```

Tests live in `tests/unit/mod.rs` (declares submodules), each in `tests/unit/<name>_tests.rs`.
Shared helpers: `crate::common::{board_from_fen, positions}` (in `tests/common/mod.rs`).

## MCTS Node Conventions

- `terminal_or_mate_value`: from **STM's perspective** at that node (−1.0 = STM loses)
- `total_value`: accumulated from **parent's perspective** (negated during backprop)
- Q from parent's view: `-(child.total_value / child.visits)`
- `nn_value`: final combined value after tanh; `v_logit`: raw NN positional logit (unbounded)
- `is_terminal`: set for checkmate, stalemate, and KOTH wins — node is never expanded

## Critical Implementation Details

- **Incremental Zobrist hashing** in `make_move.rs` — XOR updates, not full recompute
- When modifying `apply_move_to_board`: rook captures must update **opponent** castling rights too
- `ZobristKeys` accessors: `piece_key()`, `castling_key()`, `en_passant_key()`, `side_to_move_key()`
- PestoEval is used by SimpleAgent/alpha-beta only, **not** by MCTS
- mate_search is checks-only: test positions must have forced mates via all-check sequences
- When crafting checkmate test FENs: ensure capturing the checking piece is NOT legal (it must be protected)

## Python Training Pipeline

Located in `python/`. AlphaZero-style loop: self-play → replay buffer → train → export → evaluate → gate.
Neural net: OracleNet (SE-ResNet, 17×8×8 input, dual policy+value heads).

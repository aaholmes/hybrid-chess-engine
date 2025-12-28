# Testing Guide for Caissawary

Caissawary employs a comprehensive testing strategy ensuring correctness, stability, and performance. The test suite is divided into several categories targeting different layers of the engine.

## Quick Start

To run the full test suite (Unit, Integration, Property, and Regression):

```bash
./scripts/test.sh
```

To run only standard Cargo tests:

```bash
cargo test
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)
Focus on individual components in isolation.
- **Board:** FEN parsing, state representation, castling rights.
- **Move Generation:** Validity of moves, pseudo-legal vs legal generation.
- **Node:** MCTS node value logic, terminal state handling.
- **Selection:** UCB/PUCT calculations.

### 2. Integration Tests (`tests/integration/`)
Test the interaction between subsystems, particularly the MCTS search pipeline.
- **Mate Search:** Verifies the engine finds mates in complex positions.
- **Tactical Priority:** Ensures tactical moves (captures, checks) are prioritized.
- **Neural Integration:** Tests the flow between the search tree and (mocked) inference server.

### 3. Property Tests (`tests/property/`)
Uses `proptest` to generate random inputs and verify invariants.
- **Legal Moves:** Random positions are generated to ensure `generate_legal_moves` never produces illegal states.
- **Value Domains:** Verifies evaluations stay within valid bounds (e.g., tanh domain [-1, 1]).

### 4. Regression Tests (`tests/regression/`)
Target specific bugs found during development to ensure they do not reoccur.
- **Stack Overflow:** Verifies fix for `BoardStack` history handling.
- **Castling Rules:** Ensures castling is correctly blocked when traversing check.

### 5. Perft Tests (`tests/perft_tests.rs`)
Performance and correctness tests for move generation. These walk the game tree to a fixed depth and compare the leaf node count against known correct values.

```bash
cargo test --test perft_tests
```

## Running Specific Tests

To run a specific test file or test case:

```bash
# Run only regression tests
cargo test --test regression_tests

# Run a specific test case
cargo test test_castling_blocked_by_check
```

## Continuous Integration
The `scripts/test.sh` script is the entry point for CI pipelines. It executes tests in a specific order (Unit -> Integration -> Property) to fail fast on fundamental errors.

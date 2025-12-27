# Caissawary Chess Engine (formerly Kingfisher)
## A Tactics-Enhanced Hybrid MCTS Engine with State-Dependent Search Logic

Caissawary is a chess engine that combines the strategic guidance of a modern Monte Carlo Tree Search (MCTS) with the ruthless tactical precision of classical search. Its unique, state-dependent search algorithm prioritizes forcing moves and minimizes expensive neural network computations to create a brutally efficient and tactically sharp engine.

![Caissawary Logo](Caissawary.png)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange)](https://rustup.rs/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## The Name: Caissawary
Like the engine itself, the name Caissawary is also a hybrid:

- **Caïssa**: The mythical goddess of chess, representing the engine's strategic intelligence and artistry.
- **Cassowary**: A large, formidable, and famously aggressive bird, representing the engine's raw tactical power and speed.

## Core Architecture
Caissawary's intelligence stems from how it handles each node during an MCTS traversal. Instead of a single, uniform approach, its behavior adapts based on the node's state, ensuring that cheap, powerful analysis is always performed before expensive strategic evaluation.

### The MCTS Node Handling Flow
When the MCTS search selects a node, its state determines the next action:

#### 1. If the node is a new LEAF (never visited):
It is evaluated immediately to determine its value.

- **Tier 1 (Mate Search)**: A fast, parallel mate search is run. If a mate is found, this becomes the node's value.
- **Tier 2 (Quiescence Eval)**: If no mate is found, a tactical quiescence search is run to get a stable, accurate evaluation score. This score becomes the leaf's value, which is then backpropagated.

#### 2. If the node is INTERNAL with unexplored TACTICAL moves:
The engine is forced to explore a tactical move first.

- A simple heuristic (e.g., MVV-LVA) selects the next capture or promotion to analyze. 
- Quiet moves are ignored until all tactical options at this node have been tried.

#### 3. If the node is INTERNAL with only QUIET moves left:
The engine engages the powerful ResNet policy network with a "lazy evaluation" strategy.

- **First Visit**: The policy network is called exactly once to compute and store the policy priors for all available quiet moves.
- **Subsequent Visits**: The standard UCB1 formula is used to select a move, using the already-stored policy priors without needing to call the network again.

## Tier 1: Parallel Portfolio Mate Search
To find checkmates with maximum speed, the Tier 1 search utilizes **Rayon** to execute a parallel portfolio of three specialized searches that run concurrently against a shared **Atomic Node Budget**. This ensures that the most efficient algorithm for the specific type of mate finds it first, terminating the others immediately.

### Default Behavior
1.  **Sanity Check:** The engine first performs a quick, sequential check for any exhaustive Mate-in-2 or Checks-Only Mate-in-3 to instantly catch trivial wins.
2.  **Portfolio Launch:** If no shallow mate is found, the parallel portfolio is launched.

### The Portfolio Strategies
1.  **Search A: "The Spearhead" (Checks-Only)**
    *   **Constraint**: The side to move can *only* play checking moves.
    *   **Behavior**: Reaches immense depths (e.g., Mate-in-15+) because the branching factor is tiny. It finds long, forcing "check-check-check-mate" sequences that standard searches miss due to horizon effects.

2.  **Search B: "The Flanker" (One Quiet Move)**
    *   **Constraint**: Allows checks, but permits **exactly one** quiet (non-checking) move in the entire variation.
    *   **Behavior**: Finds mates requiring a setup move (e.g., blocking an escape square or a quiet sacrifice) followed by a forced sequence. This is computationally more expensive than the Spearhead but sharper than the Guardsman.

3.  **Search C: "The Guardsman" (Exhaustive)**
    *   **Constraint**: No constraints.
    *   **Behavior**: Standard Alpha-Beta search. It guarantees finding *any* mate within its depth limit (e.g., Mate-in-5) but is the shallowest of the three.

## Tier 2: Quiescence Search for Leaf Evaluation
When the MCTS traversal reaches a new leaf node and the Tier 1 search does not find a mate, the engine must still produce a robust evaluation for that position. This is the role of the Tier 2 Quiescence Search.

Instead of relying on a potentially noisy value from a neural network in a sharp position, this search resolves all immediate tactical possibilities to arrive at a stable, "quiet" position to evaluate.

- **Process**: The search expands tactical moves—primarily captures, promotions, and pressing checks—and ignores quiet moves. It continues until no more tactical moves are available.
- **Evaluation**: The final, quiet position is then scored by a classical evaluation function (Pesto) or the Neural Network (if enabled).
- **Purpose**: This process avoids the classic problem of a fixed-depth search mis-evaluating a position in the middle of a capture sequence. The resulting score is a much more reliable measure of the leaf node's true value, which is then backpropagated up the MCTS tree.

## Tier 3: Neural Network Policy (Optional)
The engine supports a **Hybrid** mode where strategic evaluation is handled by a PyTorch-trained Neural Network.

- **Architecture:** ResNet-style (or custom) architecture defined in Python.
- **Inference:** Uses **tch-rs** (LibTorch bindings) for high-performance inference within the Rust engine.
- **Lazy Evaluation:** The network is queried only when tactical resolution (Tier 1 & 2) fails to determine a clear result, saving precious GPU/CPU cycles for deep strategic thinking.

> **Note:** Neural network support is optional. Compile with `cargo build --features neural` to enable it. You must have a compatible LibTorch installed or let `tch-rs` download one.

## Training Philosophy
Caissawary is designed for high learning efficiency, making it feasible to train without nation-state-level resources.

- **Supervised Pre-training**: The recommended approach is to begin with supervised learning. The ResNet policy and the fast evaluation function should be pre-trained on a large corpus of high-quality human games. This bootstraps the engine with a strong foundation of strategic and positional knowledge.

- **Efficient Reinforcement Learning**: During subsequent self-play (RL), the engine's learning is accelerated. The built-in tactical search (Tiers 1 and 2) acts as a powerful "inductive bias," preventing the engine from making simple tactical blunders. This provides a cleaner, more focused training signal to the neural networks, allowing them to learn high-level strategy far more effectively than a "blank slate" MCTS architecture.

## Configuration
The node budgets for the tactical searches and other key parameters are designed to be configurable.

```rust
pub struct CaissawaryConfig {
    pub max_iterations: u32,
    pub time_limit: Duration,
    pub exploration_constant: f64,
    
    // Node budget for the parallel mate search at each node
    pub mate_search_nodes: u32,
    
    // Node budget for the quiescence search at each leaf
    pub quiescence_nodes: u32,
}
```

## Technical Stack
- **Core Logic**: Rust, for its performance, memory safety, and concurrency.
- **Parallelism**: **Rayon** for data parallelism in the mate search portfolio.
- **Neural Networks**: **PyTorch** (in Python) for training; **tch-rs** (LibTorch) for Rust inference.
- **Board Representation**: Bitboards, for highly efficient move generation and position manipulation.

## Building and Running

### Prerequisites
First, ensure you have the Rust toolchain installed.

```bash
# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

For the neural network components (optional), you will also need Python and PyTorch.

```bash
# Install Python dependencies
pip install torch numpy python-chess
```

### Build
Clone the repository and build the optimized release binary:

```bash
git clone https://github.com/aaholmes/caissawary.git
cd caissawary

# Standard Build (Tactical MCTS only)
cargo build --release

# Hybrid Build (With Neural Network support)
# Requires LibTorch. Automatic download may happen.
cargo build --release --features neural
```

### Usage
The primary binary is a UCI-compliant engine, suitable for use in any standard chess GUI like Arena, Cute Chess, or BanksiaGUI.

```bash
# Run the engine in UCI mode
./target/release/kingfisher
```
(Type `uci` to verify connection)

### Self-Play Data Generation
To generate training data for the neural network, use the `self_play` binary. This runs parallel games where the engine plays against itself.

```bash
# Generate 100 games with 800 simulations per move, saving to 'data/'
cargo run --release --bin self_play -- 100 800 data
```

## Testing and Benchmarking
The project includes a comprehensive suite of tests and benchmarks to validate functionality and performance.

```bash
# Run all unit and integration tests
cargo test

# Run a specific benchmark for mate-finding performance
cargo run --release --bin mate_benchmark
```

## Binary Targets
The crate is organized to produce several distinct binaries for different tasks:

- **caissawary**: The main UCI chess engine.
- **benchmark**: A suite for performance testing, measuring nodes-per-second and puzzle-solving speed.
- **self_play**: A high-throughput data generation tool that plays games against itself to create training datasets for the neural network.

## References
The architecture of Caissawary is inspired by decades of research in computer chess and artificial intelligence. Key influences include:

- Silver, D. et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- Campbell, M. et al. (2002). "Deep Blue"
- The Stockfish Engine and the NNUE architecture.

## License
This project is licensed under the terms of the MIT License. Please see the LICENSE file for details.

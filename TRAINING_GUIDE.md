# Design Doc: Self-Play Reinforcement Learning Loop

## 1. Objective
To demonstrate the superior training efficiency of the **Caissawary** (Hybrid MCTS + Mate Search) architecture compared to standard AlphaZero. We aim to create a fully automated pipeline that iteratively improves the neural network through self-play, leveraging the engine's tactical priors to accelerate learning.

## 2. Core Hypothesis
Standard AlphaZero starts from random play and must "learn" basic tactics (like mate-in-1) through trial and error, which is computationally expensive.
**Caissawary Hypothesis:** By injecting "Tactical Priors" (Mate Search + Tier 2 Tactics) into the MCTS:
1.  The engine plays tactically valid chess from Generation 0.
2.  The neural network receives cleaner, higher-quality training data (fewer random blunders).
3.  **Result:** The engine reaches a respectable Elo with significantly fewer training games/compute than AlphaZero.

## 3. Architecture

### A. The Loop (Python Controller)
A master Python script (`python/rl_loop.py`) orchestrates the cycle:
1.  **Generation Phase:** Call Rust engine to play `N` self-play games.
2.  **Training Phase:** Train PyTorch model on the generated games.
3.  **Evaluation Phase:** Play matches between `New Model` vs `Best Model`.
4.  **Promotion:** If `New` > `Best` (win rate > 55%), replace `Best`.

### B. Rust Self-Play Binary (`src/bin/self_play.rs`)
A specialized binary focused on high-throughput data generation.
*   **Input:** Path to `model.pt`, number of games, simulations per move.
*   **Concurrency:** Runs multiple games in parallel threads (using Rayon).
*   **Output:** `training_data_{gen}.json` containing a list of states, each with:
    *   **FEN (Board State):** The position in algebraic notation.
    *   **MCTS Policy Target (`pi`):** The visit counts for each legal move from the MCTS search, representing the "best move" according to the search. (Normalized to a probability distribution).
    *   **Game Outcome Value Target (`z`):** The final outcome of the game (+1 for win, 0 for draw, -1 for loss) from the perspective of the player to move in that specific FEN. This value is backpropagated to all positions in the game.

### C. Training Pipeline (`python/train_gen.py`)
*   **Input:** `training_data_{gen}.json`.
*   **Model:** Uses `MinimalViableNet` (7-layer ResNet) for speed.
*   **Output:** `model_candidate.pt`.

## 4. Implementation Steps & Estimates

### Phase 1: Rust Data Generation (Estimated: 2 Hours)
**Goal:** A binary that plays games and outputs training data.
1.  **`DataPoint` Struct:** Define serializable struct for FEN, Policy, Value.
2.  **Self-Play Logic:**
    *   Load `NeuralNetPolicy`.
    *   For each move in a game:
        *   Run `tactical_mcts_search` for a fixed number of simulations (e.g., 800).
        *   Store `(FEN, MCTS Visit Counts)` for the current position.
        *   Select a move based on `MCTS Visit Counts` (with temperature exploration).
        *   Play the move on the board.
    *   When the game ends:
        *   Determine the game outcome (win, loss, draw).
        *   Backpropagate this outcome to all stored positions in the game, setting it as the **Value Target (`z`)**.
3.  **Parallel Execution:** Use Rayon to play ~10 games concurrently.
4.  **Output:** Save to JSON/CSV.

### Phase 2: Python Training Integration (Estimated: 1.5 Hours)
**Goal:** Train the model on the Rust-generated data.
1.  **Dataset Class:** Create a PyTorch `Dataset` that reads the JSON output.
2.  **Training Script:**
    *   Load `model.pt`.
    *   Train on new data:
        *   **Policy Loss:** Optimize network's policy head to match `MCTS Policy Target` (visit counts).
        *   **Value Loss:** Optimize network's value head to match `Game Outcome Value Target`.
    *   Export new `model_new.pt`.

### Phase 3: The Loop Controller (Estimated: 1 Hour)
**Goal:** Automate the cycle.
1.  **`rl_loop.py`:**
    *   Loop `Generation` 1 to 50.
    *   Run `cargo run --bin self_play ...`.
    *   Run `python train_gen.py ...`.
    *   (Optional) Run evaluation match.
    *   Update `latest_model.pt`.

### Phase 4: Visualization (Estimated: 0.5 Hours)
**Goal:** Show the results.
1.  **Elo Tracker:** Simple CSV log of `Generation, WinRate, EstElo`.
2.  **Plotting:** Matplotlib script to graph improvement.

## 5. Execution Order

1.  **Create `src/bin/self_play.rs`:** This is the hardest part (serialization, extracting MCTS internals).
2.  **Create `python/train_gen.py`:** Standard PyTorch boilerplate.
3.  **Create `python/rl_loop.py`:** Glue code.
4.  **Run & Verify:** Run a 1-generation loop to verify data flows.

## 6. Training Constraints (AlphaZero Style)
To match standard RL practices:
*   **Exploration:** Use Dirichlet noise at the root node.
*   **Temperature:** First 30 moves $\tau=1$ (probabilistic), then $\tau \to 0$ (deterministic).
*   **Simulations:** Fixed 800 simulations per move.

## 7. Success Metric
If we can show that **Generation 5** beats **Generation 0** by >75%, the hypothesis is validated for a portfolio demonstration.

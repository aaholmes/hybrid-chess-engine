#!/bin/bash
# Overnight KOTH Training: Pure AlphaZero vs Caissawary (3-tier MCTS)
# Run two back-to-back 6-hour training sessions and generate comparison plots.

set -euo pipefail

# --- Configuration ---
RUN_DIR="runs/$(date +%Y%m%d_%H%M%S)"
BASELINE_DIR="$RUN_DIR/baseline"
CAISSAWARY_DIR="$RUN_DIR/caissawary"
PLOTS_DIR="$RUN_DIR/plots"

# Shared hyperparameters
SIMS=200
GAMES=100
MINIBATCHES=500
BATCH_SIZE=64
EVAL_GAMES=40
EVAL_SIMS=200
THRESHOLD=0.52
BUFFER_CAP=200000
MAX_GEN=60
OPTIMIZER=muon
LR=0.02
LR_SCHEDULE="15000:0.01,25000:0.005"

# --- Environment ---
TORCH_LIB=$(python3 -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1

# --- Build ---
echo "Building release binary with neural feature..."
cargo build --release --features neural

# --- Create directories ---
mkdir -p "$BASELINE_DIR"/{weights,data,buffer}
mkdir -p "$CAISSAWARY_DIR"/{weights,data,buffer}
mkdir -p "$PLOTS_DIR"

COMMON_ARGS=(
    --enable-koth
    --games-per-generation "$GAMES"
    --simulations-per-move "$SIMS"
    --minibatches-per-gen "$MINIBATCHES"
    --batch-size "$BATCH_SIZE"
    --eval-games "$EVAL_GAMES"
    --eval-simulations "$EVAL_SIMS"
    --acceptance-threshold "$THRESHOLD"
    --buffer-capacity "$BUFFER_CAP"
    --max-generations "$MAX_GEN"
    --optimizer "$OPTIMIZER"
    --initial-lr "$LR"
    --lr-schedule "$LR_SCHEDULE"
    --no-resume
)

# --- Run 1: Baseline (Pure AlphaZero) ---
echo ""
echo "============================================================"
echo "  RUN 1: BASELINE (Pure AlphaZero) — no tier1, no material"
echo "============================================================"
echo ""

timeout 6h python3 python/orchestrate.py \
    "${COMMON_ARGS[@]}" \
    --disable-tier1 --disable-material \
    --weights-dir "$BASELINE_DIR/weights" \
    --data-dir "$BASELINE_DIR/data" \
    --buffer-dir "$BASELINE_DIR/buffer" \
    --log-file "$BASELINE_DIR/training_log.jsonl" \
    2>&1 | tee "$BASELINE_DIR/full_output.log" || echo "Baseline run ended (timeout or error)"

# --- Run 2: Caissawary (Full 3-tier) ---
echo ""
echo "============================================================"
echo "  RUN 2: CAISSAWARY (Full 3-tier MCTS)"
echo "============================================================"
echo ""

timeout 6h python3 python/orchestrate.py \
    "${COMMON_ARGS[@]}" \
    --weights-dir "$CAISSAWARY_DIR/weights" \
    --data-dir "$CAISSAWARY_DIR/data" \
    --buffer-dir "$CAISSAWARY_DIR/buffer" \
    --log-file "$CAISSAWARY_DIR/training_log.jsonl" \
    2>&1 | tee "$CAISSAWARY_DIR/full_output.log" || echo "Caissawary run ended (timeout or error)"

# --- Generate Plots ---
echo ""
echo "Generating comparison plots..."
python3 python/plot_training.py \
    --baseline "$BASELINE_DIR/training_log.jsonl" \
    --caissawary "$CAISSAWARY_DIR/training_log.jsonl" \
    --output "$PLOTS_DIR/"

echo ""
echo "============================================================"
echo "  DONE. Results in: $RUN_DIR"
echo "============================================================"

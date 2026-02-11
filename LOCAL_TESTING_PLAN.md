# Local Testing Plan & Cloud Preparation

## Goal

Validate the full training pipeline locally with a tiny model (2 blocks / 64 hidden, ~240K params) before renting cloud GPUs for the neurosymbolic vs vanilla MCTS comparison.

## What Local Runs Can Tell Us

### 1. Pipeline works end-to-end (already confirmed)
Both caissawary8 and caissawary8_vanilla complete self-play, training, export, and SPRT eval without crashes.

### 2. Hyperparameter tuning (2-5 gens each)
- **Learning rate**: Is 0.02 with Muon converging too fast/slow? Check training loss curves.
- **SPRT threshold**: Is elo1=10 too strict for 400 eval games? If nothing gets accepted in 5 gens, try elo1=5.
- **Sims schedule**: Does 100 sims produce useful training data or just noise?

### 3. Gen 1 baseline
After both runs finish gen 1, compare how much each improved from random. This confirms the training signal works at all.

### 4. Acceptance feasibility
If neither run accepts a model within 10 gens, lower elo1 to 5 before the cloud run. This is the single most impactful hyperparameter for the gating loop.

## Current Local Run Config

```bash
python3 -u python/orchestrate.py \
  --weights-dir runs/long_run/caissawary8/weights \
  --data-dir runs/long_run/caissawary8/data \
  --log-file runs/long_run/caissawary8/data/training_log.jsonl \
  --buffer-dir runs/long_run/caissawary8/data/buffer \
  --max-generations 50 \
  --games-per-generation 200 \
  --minibatches 5000 \
  --sims-schedule "0:100,5:200,15:400" \
  --eval-max-games 400 \
  --enable-koth \
  --num-blocks 2 --hidden-dim 64 \
  --inference-batch-size 128 \
  --game-threads 28
```

Vanilla uses the same config plus `--disable-tier1 --disable-material`.

## What Vanilla Skips

| Component | Skipped? | Guard |
|-----------|----------|-------|
| Mate search | Yes | `config.enable_tier1` |
| KOTH geometric pruning | Yes | `config.enable_tier1` |
| `forced_material_balance()` Q-search | Yes | `config.enable_material_value` |
| k * delta_M combination | Yes | value = `v_logit.tanh()` directly |
| k-head inside NN forward pass | No | Baked into TorchScript trace (~21K params, negligible) |
| Move generation, UCB selection | No | Core MCTS logic, same for both |

## Local Run Sequence

1. **caissawary8**: Let it run several gens. Check for declining loss curves and model acceptance.
2. **Stop caissawary8** after enough data (5-10 gens or first acceptance).
3. **Start caissawary8_vanilla** with same config + `--disable-tier1 --disable-material`.
4. **Compare** training_log.jsonl from both runs.

## Success Criteria (Ready for Cloud)

- [ ] Both runs produce declining loss curves
- [ ] At least one model gets accepted within 10 gens (or elo1 lowered to 5)
- [ ] training_log.jsonl contains all needed metrics (policy_loss, value_loss, win_rate, etc.)
- [ ] No crashes or silent failures in either config

## Cloud Run Design

### Hardware
A100 or H100 GPU. Full model (6 blocks / 128 hidden, ~2M params).

### Two parallel runs

| Run | Flags | What It Tests |
|-----|-------|--------------|
| **Full system** | `--enable-koth` | Three-tier MCTS (neurosymbolic) |
| **Vanilla** | `--enable-koth --disable-tier1 --disable-material` | Pure AlphaZero-style MCTS |

### Recommended cloud config
```bash
--max-generations 50
--games-per-generation 200
--minibatches 5000
--sims-schedule "0:200,10:400,25:800"
--eval-max-games 400
--enable-koth
--num-blocks 6 --hidden-dim 128
```

### Post-training: Elo tournament
Round-robin between all accepted models from both runs using `evaluate_models` with fixed config (800 sims, --enable-koth). Compute Elo ratings with BayesElo or Ordo. Plot Elo vs generation for both runs.

### Key metrics to compare

| Metric | Source | What It Shows |
|--------|--------|--------------|
| Elo vs generation | Tournament results | Learning speed & final strength |
| Acceptance rate | training_log.jsonl | Convergence rate |
| Policy loss curve | training_log.jsonl | Move prediction quality |
| Value loss curve | training_log.jsonl | Position evaluation quality |
| Game length | Self-play logs | Decisiveness |
| Wall-clock time/gen | orchestrate.log | Efficiency |

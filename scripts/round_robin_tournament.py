#!/usr/bin/env python3
"""Round-robin tournament between tiered and vanilla models.

Each model plays with its training search configuration:
- Tiered models: tiers enabled (default)
- Vanilla models: --candidate-disable-tier1 --candidate-disable-material (or --current-)

10 models from the first ~20 generations of each run.
45 pairings, played in interleaved batches (default 10 games per batch)
so preliminary ratings are available early.
"""

import argparse
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import time

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(ROOT, "target", "release", "evaluate_models")

TIERED_DIR = os.path.join(ROOT, "runs", "long_run", "scaleup_2m_tiered_propgreedy", "weights")
VANILLA_DIR = os.path.join(ROOT, "runs", "long_run", "scaleup_2m_vanilla_propgreedy", "weights")

# Model definitions: (name, path, type)
MODELS = [
    ("tiered_gen0",  os.path.join(TIERED_DIR, "gen_0.pt"),  "tiered"),
    ("tiered_gen1",  os.path.join(TIERED_DIR, "gen_1.pt"),  "tiered"),
    ("tiered_gen4",  os.path.join(TIERED_DIR, "gen_4.pt"),  "tiered"),
    ("tiered_gen17", os.path.join(TIERED_DIR, "gen_17.pt"), "tiered"),
    ("vanilla_gen0",  os.path.join(VANILLA_DIR, "gen_0.pt"),  "vanilla"),
    ("vanilla_gen2",  os.path.join(VANILLA_DIR, "gen_2.pt"),  "vanilla"),
    ("vanilla_gen6",  os.path.join(VANILLA_DIR, "gen_6.pt"),  "vanilla"),
    ("vanilla_gen9",  os.path.join(VANILLA_DIR, "gen_9.pt"),  "vanilla"),
    ("vanilla_gen13", os.path.join(VANILLA_DIR, "gen_13.pt"), "vanilla"),
    ("vanilla_gen18", os.path.join(VANILLA_DIR, "gen_18.pt"), "vanilla"),
]

RESULTS_DIR = os.path.join(ROOT, "runs", "tournaments")


def build_cmd(cand_path, curr_path, cand_type, curr_type, num_games, simulations,
              batch_size, threads, explore_base, seed_offset):
    """Build the evaluate_models command with per-side tier flags."""
    cmd = [
        BINARY,
        cand_path, curr_path,
        str(num_games), str(simulations),
        "--enable-koth",
        "--no-save-training-data",
        "--explore-base", str(explore_base),
        "--batch-size", str(batch_size),
        "--threads", str(threads),
        "--seed-offset", str(seed_offset),
    ]

    # Per-side flags based on model type
    if cand_type == "vanilla":
        cmd += ["--candidate-disable-tier1", "--candidate-disable-material"]
    if curr_type == "vanilla":
        cmd += ["--current-disable-tier1", "--current-disable-material"]

    return cmd


def parse_result(stdout):
    """Parse WINS=X LOSSES=Y DRAWS=Z from stdout."""
    m = re.search(r"WINS=(\d+)\s+LOSSES=(\d+)\s+DRAWS=(\d+)", stdout)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def run_batch(cand_name, cand_path, cand_type, curr_name, curr_path, curr_type,
              num_games, simulations, batch_size, threads, explore_base, seed_offset):
    """Run a batch of games and return (wins, losses, draws) from candidate's perspective."""
    cmd = build_cmd(cand_path, curr_path, cand_type, curr_type,
                    num_games, simulations, batch_size, threads, explore_base, seed_offset)

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start

    # Print stderr summary (last few lines)
    if proc.stderr:
        lines = proc.stderr.strip().split("\n")
        if len(lines) > 3:
            for line in lines[-2:]:
                print(f"    {line}")
        else:
            for line in lines:
                print(f"    {line}")

    result = parse_result(proc.stdout)
    if result is None:
        print(f"    ERROR: Could not parse output: {proc.stdout[:200]}")
        print(f"    stderr: {proc.stderr[-500:]}")
        return None

    w, l, d = result
    total = w + l + d
    wr = (w + 0.5 * d) / total if total > 0 else 0.5
    print(f"    {cand_name} vs {curr_name}: +{w}-{l}={d} (WR={wr:.3f}, {elapsed:.0f}s)")
    sys.stdout.flush()
    return w, l, d


def compute_elo_mle(results, model_names, anchor_idx=0, iterations=1000):
    """Compute Elo ratings via iterative maximum likelihood estimation.

    results: dict of (i, j) -> (wins_i, wins_j, draws) where i is "candidate", j is "current"
    anchor_idx: index of the model anchored at Elo 0
    """
    n = len(model_names)
    elos = [0.0] * n

    for _ in range(iterations):
        for i in range(n):
            if i == anchor_idx:
                continue

            numerator = 0.0
            denominator = 0.0

            for j in range(n):
                if i == j:
                    continue

                # Get results between i and j
                if (i, j) in results:
                    wi, wj, d = results[(i, j)]
                elif (j, i) in results:
                    wj, wi, d = results[(j, i)]
                else:
                    continue

                # Score for player i against j
                score_i = wi + 0.5 * d
                total = wi + wj + d
                if total == 0:
                    continue

                # Expected score based on current Elo estimates
                expected = 1.0 / (1.0 + 10.0 ** ((elos[j] - elos[i]) / 400.0))

                numerator += score_i
                denominator += total * expected

            if denominator > 0:
                # Update: Elo_i += 400 * log10(numerator / denominator)
                elos[i] += 400.0 * math.log10(numerator / denominator)

    # Re-anchor
    anchor_elo = elos[anchor_idx]
    elos = [e - anchor_elo for e in elos]

    return elos


def print_ratings(model_names, elos, results, games_per_pair):
    """Print a compact sorted Elo table."""
    n = len(model_names)
    ranked = sorted(range(n), key=lambda i: elos[i], reverse=True)

    print(f"\n  {'Rank':<5} {'Model':<20} {'Elo':>8} {'W':>5} {'L':>5} {'D':>5} {'WR':>7} {'Games':>6}")
    print(f"  {'-'*62}")

    for rank, i in enumerate(ranked):
        total_w, total_l, total_d = 0, 0, 0
        for j in range(n):
            if i == j:
                continue
            if (i, j) in results:
                w, l, d = results[(i, j)]
                total_w += w
                total_l += l
                total_d += d
            elif (j, i) in results:
                l2, w2, d2 = results[(j, i)]
                total_w += w2
                total_l += l2
                total_d += d2
        total = total_w + total_l + total_d
        wr = (total_w + 0.5 * total_d) / total if total > 0 else 0.5
        print(f"  {rank+1:<5} {model_names[i]:<20} {elos[i]:>+8.1f} {total_w:>5} {total_l:>5} {total_d:>5} {wr:>7.3f} {total:>6}")


def print_cross_table(model_names, elos, results):
    """Print full cross-table."""
    n = len(model_names)
    ranked = sorted(range(n), key=lambda i: elos[i], reverse=True)

    print(f"\n  CROSS-TABLE (win rate from row's perspective)")

    # Header
    short_names = [model_names[i][:8] for i in ranked]
    header = f"  {'':>16} " + " ".join(f"{s:>8}" for s in short_names)
    print(header)
    print(f"  {'-'*(16 + 9 * n)}")

    for i in ranked:
        row = f"  {model_names[i]:>16} "
        for j in ranked:
            if i == j:
                row += f"{'---':>8}"
                continue
            if (i, j) in results:
                w, l, d = results[(i, j)]
            elif (j, i) in results:
                l, w, d = results[(j, i)]
            else:
                row += f"{'?':>8}"
                continue
            total = w + l + d
            wr = (w + 0.5 * d) / total if total > 0 else 0.5
            row += f"{wr:>8.3f}"
        print(row)


def save_results(model_names, elos, results, games_played, output_dir):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "models": model_names,
        "elos": elos,
        "results": {f"{i},{j}": list(v) for (i, j), v in results.items()},
        "games_played": {f"{i},{j}": g for (i, j), g in games_played.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(output_dir, "round_robin_results.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_partial_results(output_dir):
    """Load partial results from a previous run."""
    path = os.path.join(output_dir, "round_robin_results.json")
    if not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    results = {}
    for key, val in data.get("results", {}).items():
        i, j = map(int, key.split(","))
        results[(i, j)] = tuple(val)
    games_played = {}
    for key, val in data.get("games_played", {}).items():
        i, j = map(int, key.split(","))
        games_played[(i, j)] = val
    return results, games_played


def bootstrap_consecutive_ci(results, model_names, n_bootstrap=200):
    """Bootstrap consecutive-rank Elo gaps to find the most uncertain pairing.

    Returns list of (idx_higher, idx_lower, gap_mean, gap_ci_width, games_played)
    sorted by gap_ci_width descending (most uncertain first).
    """
    n = len(model_names)

    # Build per-pair game counts for resampling
    pair_data = {}
    for (i, j), (w, l, d) in results.items():
        total = w + l + d
        if total > 0:
            pair_data[(i, j)] = (w, l, d, total)

    if not pair_data:
        return []

    # Current MLE ranking to define consecutive pairs
    elos = compute_elo_mle(results, model_names)
    ranked = sorted(range(n), key=lambda i: elos[i], reverse=True)

    # Bootstrap: resample results, compute Elo, record consecutive gaps
    gap_samples = {k: [] for k in range(n - 1)}  # gap_samples[k] = list of gap values

    for _ in range(n_bootstrap):
        # Resample results
        resampled = {}
        for (i, j), (w, l, d, total) in pair_data.items():
            outcomes = random.choices(['w', 'l', 'd'], weights=[w, l, d], k=total)
            rw = outcomes.count('w')
            rl = outcomes.count('l')
            rd = outcomes.count('d')
            resampled[(i, j)] = (rw, rl, rd)

        # Compute Elo from resampled
        boot_elos = compute_elo_mle(resampled, model_names)
        boot_ranked = sorted(range(n), key=lambda i: boot_elos[i], reverse=True)

        # Record consecutive gaps (using current ranking order, not bootstrap ranking)
        for k in range(n - 1):
            higher = ranked[k]
            lower = ranked[k + 1]
            gap = boot_elos[higher] - boot_elos[lower]
            gap_samples[k].append(gap)

    # Compute mean and CI width for each consecutive gap
    consecutive_gaps = []
    for k in range(n - 1):
        samples = gap_samples[k]
        if not samples:
            continue
        mean_gap = sum(samples) / len(samples)
        std_gap = (sum((s - mean_gap) ** 2 for s in samples) / len(samples)) ** 0.5
        ci_width = 1.96 * std_gap  # 95% CI half-width

        higher = ranked[k]
        lower = ranked[k + 1]

        # Count games between this pair
        if (higher, lower) in results:
            gp = sum(results[(higher, lower)])
        elif (lower, higher) in results:
            gp = sum(results[(lower, higher)])
        else:
            gp = 0

        consecutive_gaps.append((higher, lower, mean_gap, ci_width, gp))

    # Sort by CI width descending (most uncertain first)
    consecutive_gaps.sort(key=lambda x: x[3], reverse=True)
    return consecutive_gaps


def print_ci_table(consecutive_gaps, model_names, elos, focus_pair=None):
    """Print the consecutive Elo gap CI table."""
    ranked = sorted(range(len(model_names)), key=lambda i: elos[i], reverse=True)
    rank_of = {idx: r + 1 for r, idx in enumerate(ranked)}

    print(f"\n  {'Gap':<8} {'Higher':<18} {'Lower':<18} {'Elo Gap':>8} {'CI±':>7} {'Games':>6}")
    print(f"  {'-'*69}")

    for higher, lower, gap_mean, ci_width, gp in consecutive_gaps:
        marker = "  <-- NEXT" if focus_pair and (higher, lower) == focus_pair else ""
        rh, rl = rank_of[higher], rank_of[lower]
        print(f"  #{rh}-#{rl}  {model_names[higher]:<18} {model_names[lower]:<18} "
              f"{gap_mean:>+8.1f} {ci_width:>6.1f} {gp:>6}{marker}")


def run_adaptive_phase(model_names, results, games_played, args, output_dir, tournament_start):
    """Focus games on consecutive-rank pairs with widest CI."""
    n = len(model_names)

    while True:
        # Check total games limit
        total_games = sum(sum(v) for v in results.values())
        if total_games >= args.max_total_games:
            print(f"\n  Reached max total games ({total_games}/{args.max_total_games}). Stopping.")
            break

        # Bootstrap to find most uncertain consecutive pair
        consecutive_gaps = bootstrap_consecutive_ci(results, model_names)
        if not consecutive_gaps:
            print("  No consecutive gaps to analyze.")
            break

        # Check stopping condition: max CI < target
        max_ci = consecutive_gaps[0][3]
        if max_ci < args.ci_target:
            print(f"\n  All consecutive CIs below target ({max_ci:.1f} < {args.ci_target}). Stopping.")
            break

        # Select pair with widest CI
        higher, lower, gap_mean, ci_width, gp = consecutive_gaps[0]

        # Print CI table
        elos = compute_elo_mle(results, model_names)
        elapsed = time.time() - tournament_start
        print(f"\n{'#'*70}")
        print(f"  ADAPTIVE FOCUS — {total_games} total games, {elapsed/60:.0f}min")
        print(f"{'#'*70}")
        print_ratings(model_names, elos, results, games_played)
        print_ci_table(consecutive_gaps, model_names, elos, focus_pair=(higher, lower))

        # Determine which is candidate/current in results dict
        # We need a canonical key — use (min, max) ordering matching pairings
        i, j = (min(higher, lower), max(higher, lower))
        played = games_played.get((i, j), 0)
        seed_offset = i * 10000 + j * 100 + played

        cand_name, cand_path, cand_type = MODELS[i]
        curr_name, curr_path, curr_type = MODELS[j]

        batch_games = min(args.batch, args.max_total_games - total_games)
        if batch_games <= 0:
            break

        print(f"\n  Playing {batch_games} games: {cand_name} vs {curr_name}")
        sys.stdout.flush()

        result = run_batch(
            cand_name, cand_path, cand_type,
            curr_name, curr_path, curr_type,
            batch_games, args.sims, args.batch_size, args.threads,
            args.explore_base, seed_offset,
        )

        if result is not None:
            w, l, d = result
            if (i, j) in results:
                pw, pl, pd = results[(i, j)]
                results[(i, j)] = (pw + w, pl + l, pd + d)
            else:
                results[(i, j)] = (w, l, d)
            games_played[(i, j)] = played + batch_games

            elos = compute_elo_mle(results, model_names)
            save_results(model_names, elos, results, games_played, output_dir)
        else:
            print(f"    FAILED: {cand_name} vs {curr_name}")

        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Round-robin tournament")
    parser.add_argument("--games", type=int, default=200, help="Total games per match (non-adaptive mode)")
    parser.add_argument("--batch", type=int, default=10, help="Games per batch before rotating to next pairing")
    parser.add_argument("--sims", type=int, default=200, help="Simulations per move")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--threads", type=int, default=28, help="Game threads")
    parser.add_argument("--explore-base", type=float, default=1.0, help="Explore base (1.0=always proportional)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from partial results")
    # Adaptive mode arguments
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive mode: seed then focus on uncertain pairs")
    parser.add_argument("--seed-games", type=int, default=10, help="Games per pair in seed phase (adaptive mode)")
    parser.add_argument("--ci-target", type=float, default=50.0, help="Stop when max consecutive CI < this (adaptive mode)")
    parser.add_argument("--max-total-games", type=int, default=9000, help="Max total games across all pairs (adaptive mode)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, "round_robin_10model")
    os.makedirs(output_dir, exist_ok=True)

    model_names = [m[0] for m in MODELS]
    n = len(model_names)

    # In adaptive mode, seed phase uses --seed-games; otherwise use --games
    seed_games = args.seed_games if args.adaptive else args.games
    num_batches = (seed_games + args.batch - 1) // args.batch  # ceiling division

    # Verify all model files exist
    for name, path, mtype in MODELS:
        if not os.path.exists(path):
            print(f"ERROR: Model file not found: {path} ({name})")
            sys.exit(1)

    # Verify binary exists
    if not os.path.exists(BINARY):
        print(f"ERROR: Binary not found: {BINARY}")
        print("Run: cargo build --release --bin evaluate_models --features neural")
        sys.exit(1)

    # Load partial results if resuming
    if args.resume:
        results, games_played = load_partial_results(output_dir)
    else:
        results, games_played = {}, {}

    # All pairings (i < j to avoid duplicates)
    pairings = list(itertools.combinations(range(n), 2))
    total_pairings = len(pairings)

    if args.adaptive:
        print(f"Tournament (adaptive): {n} models, {total_pairings} pairings")
        print(f"Seed phase: {seed_games} games/pair, then focus on uncertain consecutive pairs")
        print(f"CI target: {args.ci_target} Elo, max total games: {args.max_total_games}")
    else:
        print(f"Tournament: {n} models, {total_pairings} pairings, {args.games} games each")
        print(f"Batch size: {args.batch} games/pairing, {num_batches} rounds to complete")
    print(f"Models: {', '.join(model_names)}")
    if games_played:
        min_g = min(games_played.get(p, 0) for p in pairings)
        max_g = max(games_played.get(p, 0) for p in pairings)
        print(f"Resuming: {min_g}-{max_g} games played per pairing")
    print()

    tournament_start = time.time()

    # --- Seed phase (interleaved round-robin) ---
    for round_num in range(num_batches):
        remaining = [(i, j) for (i, j) in pairings if games_played.get((i, j), 0) < seed_games]
        if not remaining:
            break

        games_so_far = min(games_played.get(p, 0) for p in pairings)
        games_target = min(games_so_far + args.batch, seed_games)

        phase_label = "SEED" if args.adaptive else "ROUND"
        print(f"\n{'#'*70}")
        print(f"  {phase_label} {round_num + 1}/{num_batches} — playing to {games_target} games/pairing")
        print(f"{'#'*70}")
        sys.stdout.flush()

        for i, j in pairings:
            played = games_played.get((i, j), 0)
            if played >= games_target:
                continue

            batch_games = games_target - played
            cand_name, cand_path, cand_type = MODELS[i]
            curr_name, curr_path, curr_type = MODELS[j]

            seed_offset = i * 10000 + j * 100 + played

            result = run_batch(
                cand_name, cand_path, cand_type,
                curr_name, curr_path, curr_type,
                batch_games, args.sims, args.batch_size, args.threads,
                args.explore_base, seed_offset,
            )

            if result is not None:
                w, l, d = result
                if (i, j) in results:
                    pw, pl, pd = results[(i, j)]
                    results[(i, j)] = (pw + w, pl + l, pd + d)
                else:
                    results[(i, j)] = (w, l, d)
                games_played[(i, j)] = played + batch_games
            else:
                print(f"    FAILED: {cand_name} vs {curr_name}")

        # Print ratings after each round
        if results:
            elos = compute_elo_mle(results, model_names)
            total_games = sum(sum(v) for v in results.values())
            elapsed = time.time() - tournament_start
            print(f"\n  === RATINGS after {phase_label.lower()} {round_num + 1} ({total_games} total games, {elapsed/60:.0f}min) ===")
            print_ratings(model_names, elos, results, games_played)
            save_results(model_names, elos, results, games_played, output_dir)
            sys.stdout.flush()

    # --- Focus phase (adaptive only) ---
    if args.adaptive and results:
        print(f"\n\n{'='*70}")
        print(f"  ENTERING ADAPTIVE FOCUS PHASE")
        print(f"{'='*70}")
        run_adaptive_phase(model_names, results, games_played, args, output_dir, tournament_start)

    # Final results
    if results:
        elos = compute_elo_mle(results, model_names)
        elapsed = time.time() - tournament_start
        print(f"\n\n{'='*70}")
        print(f"  FINAL RESULTS ({elapsed/60:.0f} minutes)")
        print(f"{'='*70}")
        print_ratings(model_names, elos, results, games_played)
        print_cross_table(model_names, elos, results)
        if args.adaptive:
            consecutive_gaps = bootstrap_consecutive_ci(results, model_names)
            if consecutive_gaps:
                print_ci_table(consecutive_gaps, model_names, elos)
        path = save_results(model_names, elos, results, games_played, output_dir)
        print(f"\nResults saved to {path}")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()

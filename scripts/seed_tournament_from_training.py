#!/usr/bin/env python3
"""Extract eval results from training logs to pre-seed a round-robin tournament.

During training, each generation is evaluated against the current best via SPRT
(up to 800 games). For accepted generations, this produces high-quality head-to-head
results between consecutive accepted models. These results use the same search config
as the tournament, so they can be directly imported as seed data.

Only pairs where BOTH models are tournament participants are useful. Since only
accepted gens become tournament models, the useful pairs are consecutive-accepted:
gen_1 vs gen_0, gen_3 vs gen_1, gen_4 vs gen_3, etc.

Usage:
    python3 scripts/seed_tournament_from_training.py \
        --tiered-log runs/.../training_log.jsonl \
        --tiered-dir runs/.../weights \
        --vanilla-log runs/.../training_log.jsonl \
        --vanilla-dir runs/.../weights \
        --tiered-gens 0,1,4,17 \
        --vanilla-gens 0,2,6,9,13,18 \
        --output runs/tournaments/round_robin/round_robin_results.json
"""

import argparse
import json
import os
import re
import sys
import time


def parse_training_log(log_path):
    """Parse a training log and return list of entries."""
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_gen_from_path(path):
    """Extract generation number from a model path like '.../gen_5.pt'."""
    m = re.search(r'gen_(\d+)\.pt', path)
    if m:
        return int(m.group(1))
    return None


def extract_eval_pairs(entries):
    """Extract all (candidate_gen, opponent_gen, wins, losses, draws) from training log.

    Tracks the current_best before each entry to determine the opponent.
    Returns list of (candidate_gen, opponent_gen, wins, losses, draws).
    """
    pairs = []
    prev_best_gen = 0  # gen_0 is always the starting model

    for entry in entries:
        candidate_gen = entry['gen']
        opponent_gen = prev_best_gen

        wins = entry.get('eval_wins', 0)
        losses = entry.get('eval_losses', 0)
        draws = entry.get('eval_draws', 0)
        total = wins + losses + draws

        if total > 0:
            pairs.append((candidate_gen, opponent_gen, wins, losses, draws))

        # Update prev_best from current_best path
        if entry.get('accepted'):
            current_best = entry.get('current_best', '')
            gen_num = extract_gen_from_path(current_best)
            if gen_num is not None:
                prev_best_gen = gen_num
            else:
                # Fallback: accepted gen becomes the new best
                prev_best_gen = candidate_gen

    return pairs


def build_model_list(tiered_gens, vanilla_gens, tiered_dir, vanilla_dir):
    """Build the tournament model list matching round_robin_tournament.py format.

    Returns list of (name, path, type) tuples.
    """
    models = []
    for g in tiered_gens:
        models.append((f"tiered_gen{g}", os.path.join(tiered_dir, f"gen_{g}.pt"), "tiered"))
    for g in vanilla_gens:
        models.append((f"vanilla_gen{g}", os.path.join(vanilla_dir, f"gen_{g}.pt"), "vanilla"))
    return models


def seed_results(tiered_pairs, vanilla_pairs, models, tiered_gens, vanilla_gens):
    """Map eval pairs to tournament indices and build results/games_played dicts.

    Returns (results, games_played) where:
        results: {(i, j): (wins_i, wins_j, draws)} with i < j
        games_played: {(i, j): total_games}
    """
    model_names = [m[0] for m in models]

    # Build lookup: (type, gen) -> model index
    idx_of = {}
    for i, (name, _, mtype) in enumerate(models):
        m = re.search(r'gen(\d+)', name)
        if m:
            idx_of[(mtype, int(m.group(1)))] = i

    results = {}
    games_played = {}
    seeded_pairs = []

    def add_pair(mtype, candidate_gen, opponent_gen, wins, losses, draws):
        cand_idx = idx_of.get((mtype, candidate_gen))
        opp_idx = idx_of.get((mtype, opponent_gen))

        if cand_idx is None or opp_idx is None:
            return  # One or both not in tournament

        total = wins + losses + draws
        if total == 0:
            return

        # Canonical key: i < j
        i, j = min(cand_idx, opp_idx), max(cand_idx, opp_idx)

        # Determine wins from i's perspective
        if cand_idx == i:
            # candidate is the lower index
            wi, wj, d = wins, losses, draws
        else:
            # candidate is the higher index, flip
            wi, wj, d = losses, wins, draws

        # Accumulate (in case multiple entries for same pair, though unlikely)
        if (i, j) in results:
            pw, pl, pd = results[(i, j)]
            results[(i, j)] = (pw + wi, pl + wj, pd + d)
            games_played[(i, j)] += total
        else:
            results[(i, j)] = (wi, wj, d)
            games_played[(i, j)] = total

        seeded_pairs.append((mtype, candidate_gen, opponent_gen, wins, losses, draws, total))

    for cand_gen, opp_gen, w, l, d in tiered_pairs:
        add_pair("tiered", cand_gen, opp_gen, w, l, d)

    for cand_gen, opp_gen, w, l, d in vanilla_pairs:
        add_pair("vanilla", cand_gen, opp_gen, w, l, d)

    return results, games_played, seeded_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Seed round-robin tournament from training eval data")
    parser.add_argument("--tiered-log", type=str, default=None,
                        help="Path to tiered training_log.jsonl")
    parser.add_argument("--tiered-dir", type=str, default=None,
                        help="Path to tiered weights directory")
    parser.add_argument("--vanilla-log", type=str, default=None,
                        help="Path to vanilla training_log.jsonl")
    parser.add_argument("--vanilla-dir", type=str, default=None,
                        help="Path to vanilla weights directory")
    parser.add_argument("--tiered-gens", type=str, default="",
                        help="Comma-separated accepted gen numbers for tiered (e.g. 0,1,4,17)")
    parser.add_argument("--vanilla-gens", type=str, default="",
                        help="Comma-separated accepted gen numbers for vanilla (e.g. 0,2,6,9,13,18)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for round_robin_results.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be seeded without writing")
    args = parser.parse_args()

    # Parse gen lists
    tiered_gens = [int(x) for x in args.tiered_gens.split(",") if x.strip()] if args.tiered_gens else []
    vanilla_gens = [int(x) for x in args.vanilla_gens.split(",") if x.strip()] if args.vanilla_gens else []

    if not tiered_gens and not vanilla_gens:
        print("ERROR: Must specify at least one of --tiered-gens or --vanilla-gens")
        sys.exit(1)

    # Extract eval pairs from training logs
    tiered_pairs = []
    if args.tiered_log and tiered_gens:
        if not os.path.exists(args.tiered_log):
            print(f"ERROR: Tiered log not found: {args.tiered_log}")
            sys.exit(1)
        entries = parse_training_log(args.tiered_log)
        tiered_pairs = extract_eval_pairs(entries)
        print(f"Tiered log: {len(entries)} entries, {len(tiered_pairs)} eval pairs")

    vanilla_pairs = []
    if args.vanilla_log and vanilla_gens:
        if not os.path.exists(args.vanilla_log):
            print(f"ERROR: Vanilla log not found: {args.vanilla_log}")
            sys.exit(1)
        entries = parse_training_log(args.vanilla_log)
        vanilla_pairs = extract_eval_pairs(entries)
        print(f"Vanilla log: {len(entries)} entries, {len(vanilla_pairs)} eval pairs")

    # Build model list
    tiered_dir = args.tiered_dir or ""
    vanilla_dir = args.vanilla_dir or ""
    models = build_model_list(tiered_gens, vanilla_gens, tiered_dir, vanilla_dir)
    model_names = [m[0] for m in models]

    print(f"\nTournament models ({len(models)}):")
    for i, (name, path, mtype) in enumerate(models):
        exists = os.path.exists(path) if path else False
        status = "OK" if exists else "MISSING"
        print(f"  [{i}] {name} ({status})")

    # Map pairs to tournament indices
    results, games_played, seeded_pairs = seed_results(
        tiered_pairs, vanilla_pairs, models, tiered_gens, vanilla_gens)

    # Print summary
    total_seeded = sum(sum(v) for v in results.values())
    print(f"\nSeeded {len(results)} pairs with {total_seeded} total games:")
    for mtype, cand, opp, w, l, d, total in seeded_pairs:
        wr = (w + 0.5 * d) / total if total > 0 else 0.5
        print(f"  {mtype}_gen{cand} vs {mtype}_gen{opp}: +{w}-{l}={d} ({total} games, WR={wr:.3f})")

    if args.dry_run:
        print("\n[DRY RUN] Would write to:", args.output)
        return

    # Write results JSON in round_robin_tournament.py format
    # Compute placeholder Elos (will be recomputed by tournament script)
    elos = [0.0] * len(models)

    data = {
        "models": model_names,
        "elos": elos,
        "results": {f"{i},{j}": list(v) for (i, j), v in results.items()},
        "games_played": {f"{i},{j}": g for (i, j), g in games_played.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeded_from_training": True,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nWrote seeded results to: {args.output}")


if __name__ == "__main__":
    main()

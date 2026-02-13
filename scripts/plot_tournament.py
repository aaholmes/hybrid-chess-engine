#!/usr/bin/env python3
"""Compute MLE Elo ratings from tournament CSV and plot Elo vs generation."""

import csv
import math
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_pairwise_results(csv_path):
    """Load pairwise results from the tournament CSV (stops at blank line)."""
    results = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip():
                break
            model_a, model_b = row[0], row[1]
            wins_a, draws, wins_b = int(row[2]), int(row[3]), int(row[4])
            results.append((model_a, model_b, wins_a, draws, wins_b))
    return results


def mle_elo(results, iterations=2000, lr=10.0):
    """Compute MLE Elo ratings via gradient ascent on Bradley-Terry log-likelihood."""
    # Collect all model names
    names = set()
    for a, b, *_ in results:
        names.add(a)
        names.add(b)
    names = sorted(names)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    ratings = [1500.0] * n

    for _ in range(iterations):
        grad = [0.0] * n
        for model_a, model_b, wins_a, draws, wins_b in results:
            i, j = idx[model_a], idx[model_b]
            n_games = wins_a + draws + wins_b
            if n_games == 0:
                continue
            s_ij = wins_a + draws / 2.0  # score for model_a
            expected = 1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / 400.0))
            g = (math.log(10) / 400.0) * (s_ij - n_games * expected)
            grad[i] += g
            grad[j] -= g

        for k in range(n):
            ratings[k] += lr * grad[k]

        # Re-anchor mean to 1500
        mean = sum(ratings) / n
        for k in range(n):
            ratings[k] += 1500.0 - mean

    return {name: ratings[idx[name]] for name in names}


def log_likelihood(results, ratings):
    """Compute the log-likelihood of the observed results given ratings."""
    ll = 0.0
    for model_a, model_b, wins_a, draws, wins_b in results:
        n_games = wins_a + draws + wins_b
        if n_games == 0:
            continue
        s_ij = wins_a + draws / 2.0
        expected = 1.0 / (1.0 + 10.0 ** ((ratings[model_b] - ratings[model_a]) / 400.0))
        # Avoid log(0)
        expected = max(min(expected, 1.0 - 1e-10), 1e-10)
        ll += s_ij * math.log(expected) + (n_games - s_ij) * math.log(1.0 - expected)
    return ll


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "tournament_results_14way.csv"
    results = load_pairwise_results(csv_path)
    ratings = mle_elo(results)

    # Print ratings
    ranked = sorted(ratings.items(), key=lambda x: -x[1])
    print("=== MLE Elo Ratings ===")
    print(f"{'Model':<20} {'Elo':>6}")
    print("-" * 28)
    for name, elo in ranked:
        print(f"{name:<20} {elo:>+6.1f}")

    ll = log_likelihood(results, ratings)
    print(f"\nLog-likelihood: {ll:.2f}")

    # Parse generation numbers and group by run
    tiered = {}
    vanilla = {}
    for name, elo in ratings.items():
        gen = int(name.split("gen")[1])
        if name.startswith("tiered_"):
            tiered[gen] = elo
        elif name.startswith("vanilla_"):
            vanilla[gen] = elo

    # Sort by generation
    tiered_gens = sorted(tiered.keys())
    tiered_elos = [tiered[g] for g in tiered_gens]
    vanilla_gens = sorted(vanilla.keys())
    vanilla_elos = [vanilla[g] for g in vanilla_gens]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tiered_gens, tiered_elos, "o-", color="#2563eb", linewidth=2,
            markersize=7, label="Tiered (tier1 + material + KOTH)", zorder=3)
    ax.plot(vanilla_gens, vanilla_elos, "s-", color="#dc2626", linewidth=2,
            markersize=7, label="Vanilla (KOTH only)", zorder=3)

    ax.set_xlabel("Accepted Generation", fontsize=12)
    ax.set_ylabel("Elo Rating (MLE)", fontsize=12)
    ax.set_title("Caissawary: Tiered vs Vanilla MCTS Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, max(max(tiered_gens), max(vanilla_gens)) + 1)

    plt.tight_layout()
    out_path = csv_path.replace(".csv", "_elo_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

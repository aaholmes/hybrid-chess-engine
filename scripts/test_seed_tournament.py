#!/usr/bin/env python3
"""Tests for seed_tournament_from_training.py"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seed_tournament_from_training as seed


class TestExtractGenFromPath(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(seed.extract_gen_from_path("weights/gen_5.pt"), 5)

    def test_full_path(self):
        self.assertEqual(
            seed.extract_gen_from_path("runs/long_run/scaleup/weights/gen_17.pt"), 17)

    def test_no_match(self):
        self.assertIsNone(seed.extract_gen_from_path("candidate_3.pt"))

    def test_gen_zero(self):
        self.assertEqual(seed.extract_gen_from_path("gen_0.pt"), 0)


class TestExtractEvalPairs(unittest.TestCase):
    def _make_entry(self, gen, accepted, wins, losses, draws, current_best_gen):
        return {
            "gen": gen,
            "accepted": accepted,
            "eval_wins": wins,
            "eval_losses": losses,
            "eval_draws": draws,
            "eval_games_played": wins + losses + draws,
            "current_best": f"weights/gen_{current_best_gen}.pt",
        }

    def test_single_accepted(self):
        entries = [self._make_entry(1, True, 80, 14, 50, 1)]
        pairs = seed.extract_eval_pairs(entries)
        self.assertEqual(len(pairs), 1)
        cand, opp, w, l, d = pairs[0]
        self.assertEqual(cand, 1)
        self.assertEqual(opp, 0)  # gen_0 is the starting best
        self.assertEqual(w, 80)
        self.assertEqual(l, 14)
        self.assertEqual(d, 50)

    def test_rejected_then_accepted(self):
        entries = [
            self._make_entry(1, False, 50, 50, 0, 0),  # rejected, best stays gen_0
            self._make_entry(2, True, 70, 20, 10, 2),   # accepted, best becomes gen_2
        ]
        pairs = seed.extract_eval_pairs(entries)
        self.assertEqual(len(pairs), 2)
        # Gen 1 vs gen_0
        self.assertEqual(pairs[0][:2], (1, 0))
        # Gen 2 vs gen_0 (still gen_0 since gen_1 was rejected)
        self.assertEqual(pairs[1][:2], (2, 0))

    def test_consecutive_accepted(self):
        entries = [
            self._make_entry(1, True, 80, 14, 50, 1),
            self._make_entry(2, True, 60, 30, 10, 2),
        ]
        pairs = seed.extract_eval_pairs(entries)
        self.assertEqual(pairs[0][:2], (1, 0))
        self.assertEqual(pairs[1][:2], (2, 1))

    def test_skip_zero_games(self):
        entries = [self._make_entry(1, False, 0, 0, 0, 0)]
        pairs = seed.extract_eval_pairs(entries)
        self.assertEqual(len(pairs), 0)

    def test_tracks_best_through_rejections(self):
        entries = [
            self._make_entry(1, True, 80, 14, 50, 1),   # best = gen_1
            self._make_entry(2, False, 50, 50, 0, 1),   # rejected, best stays gen_1
            self._make_entry(3, False, 40, 60, 0, 1),   # rejected, best stays gen_1
            self._make_entry(4, True, 70, 20, 10, 4),   # best = gen_4
        ]
        pairs = seed.extract_eval_pairs(entries)
        # All opponents should be correct
        self.assertEqual(pairs[0][:2], (1, 0))
        self.assertEqual(pairs[1][:2], (2, 1))
        self.assertEqual(pairs[2][:2], (3, 1))
        self.assertEqual(pairs[3][:2], (4, 1))


class TestBuildModelList(unittest.TestCase):
    def test_basic(self):
        models = seed.build_model_list([0, 1], [0, 2], "/tiered", "/vanilla")
        self.assertEqual(len(models), 4)
        self.assertEqual(models[0], ("tiered_gen0", "/tiered/gen_0.pt", "tiered"))
        self.assertEqual(models[2], ("vanilla_gen0", "/vanilla/gen_0.pt", "vanilla"))

    def test_tiered_first(self):
        models = seed.build_model_list([5], [3], "/t", "/v")
        self.assertEqual(models[0][2], "tiered")
        self.assertEqual(models[1][2], "vanilla")

    def test_empty_gens(self):
        models = seed.build_model_list([], [0, 1], "", "/v")
        self.assertEqual(len(models), 2)
        self.assertTrue(all(m[2] == "vanilla" for m in models))


class TestSeedResults(unittest.TestCase):
    def test_consecutive_accepted_pair(self):
        """Consecutive accepted gens that are both in tournament should be seeded."""
        models = [
            ("tiered_gen0", "/t/gen_0.pt", "tiered"),
            ("tiered_gen1", "/t/gen_1.pt", "tiered"),
        ]
        tiered_pairs = [(1, 0, 80, 14, 50)]  # gen1 beat gen0
        results, gp, seeded = seed.seed_results(
            tiered_pairs, [], models, [0, 1], [])

        self.assertEqual(len(results), 1)
        # Key (0,1): wins_for_0=14, wins_for_1=80, draws=50
        self.assertEqual(results[(0, 1)], (14, 80, 50))
        self.assertEqual(gp[(0, 1)], 144)

    def test_non_tournament_gen_skipped(self):
        """Pairs where one gen isn't in tournament should be skipped."""
        models = [
            ("tiered_gen0", "/t/gen_0.pt", "tiered"),
            ("tiered_gen4", "/t/gen_4.pt", "tiered"),
        ]
        # gen_1 vs gen_0: gen_1 not in tournament
        tiered_pairs = [(1, 0, 80, 14, 50)]
        results, gp, seeded = seed.seed_results(
            tiered_pairs, [], models, [0, 4], [])

        self.assertEqual(len(results), 0)

    def test_both_types(self):
        """Should handle tiered and vanilla pairs."""
        models = [
            ("tiered_gen0", "/t/gen_0.pt", "tiered"),
            ("tiered_gen1", "/t/gen_1.pt", "tiered"),
            ("vanilla_gen0", "/v/gen_0.pt", "vanilla"),
            ("vanilla_gen2", "/v/gen_2.pt", "vanilla"),
        ]
        tiered_pairs = [(1, 0, 80, 14, 50)]
        vanilla_pairs = [(2, 0, 60, 30, 10)]
        results, gp, seeded = seed.seed_results(
            tiered_pairs, vanilla_pairs, models, [0, 1], [0, 2])

        self.assertEqual(len(results), 2)
        # Tiered: (0,1) = gen0 vs gen1
        self.assertIn((0, 1), results)
        # Vanilla: (2,3) = vanilla_gen0 vs vanilla_gen2
        self.assertIn((2, 3), results)

    def test_higher_index_candidate(self):
        """When candidate has higher index, W/L should be flipped in key."""
        models = [
            ("vanilla_gen0", "/v/gen_0.pt", "vanilla"),
            ("vanilla_gen2", "/v/gen_2.pt", "vanilla"),
        ]
        # gen_2 (idx 1) beat gen_0 (idx 0): W=60, L=30, D=10
        vanilla_pairs = [(2, 0, 60, 30, 10)]
        results, gp, seeded = seed.seed_results(
            [], vanilla_pairs, models, [], [0, 2])

        # Key (0,1): candidate is idx 1 (higher), so flip: wins_for_0=30, wins_for_1=60
        self.assertEqual(results[(0, 1)], (30, 60, 10))

    def test_zero_games_skipped(self):
        """Pairs with 0 total games should not be added."""
        models = [
            ("tiered_gen0", "/t/gen_0.pt", "tiered"),
            ("tiered_gen1", "/t/gen_1.pt", "tiered"),
        ]
        tiered_pairs = [(1, 0, 0, 0, 0)]
        results, gp, seeded = seed.seed_results(
            tiered_pairs, [], models, [0, 1], [])

        self.assertEqual(len(results), 0)


class TestEndToEnd(unittest.TestCase):
    def test_write_and_load(self):
        """Seed script output should be loadable by tournament script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake training log
            log_path = os.path.join(tmpdir, "training_log.jsonl")
            entries = [
                {"gen": 1, "accepted": True, "eval_wins": 80, "eval_losses": 14,
                 "eval_draws": 50, "eval_games_played": 144,
                 "current_best": "weights/gen_1.pt"},
                {"gen": 2, "accepted": True, "eval_wins": 60, "eval_losses": 30,
                 "eval_draws": 10, "eval_games_played": 100,
                 "current_best": "weights/gen_2.pt"},
            ]
            with open(log_path, 'w') as f:
                for e in entries:
                    f.write(json.dumps(e) + '\n')

            # Run seed extraction
            parsed = seed.parse_training_log(log_path)
            pairs = seed.extract_eval_pairs(parsed)

            models = seed.build_model_list([0, 1, 2], [], tmpdir, "")
            results, gp, seeded = seed.seed_results(pairs, [], models, [0, 1, 2], [])

            # Write output
            output_path = os.path.join(tmpdir, "round_robin_results.json")
            data = {
                "models": [m[0] for m in models],
                "elos": [0.0] * len(models),
                "results": {f"{i},{j}": list(v) for (i, j), v in results.items()},
                "games_played": {f"{i},{j}": g for (i, j), g in gp.items()},
            }
            with open(output_path, 'w') as f:
                json.dump(data, f)

            # Load with tournament script's loader
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import round_robin_tournament as rrt
            loaded_results, loaded_gp = rrt.load_partial_results(tmpdir)

            self.assertEqual(len(loaded_results), 2)
            # gen_1 vs gen_0: (0,1) = (14, 80, 50)
            self.assertEqual(loaded_results[(0, 1)], (14, 80, 50))
            # gen_2 vs gen_1: (1,2) = (30, 60, 10)
            self.assertEqual(loaded_results[(1, 2)], (30, 60, 10))


if __name__ == "__main__":
    unittest.main()

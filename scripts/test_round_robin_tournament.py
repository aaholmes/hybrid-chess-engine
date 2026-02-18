#!/usr/bin/env python3
"""Tests for round_robin_tournament.py"""

import json
import os
import random
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import round_robin_tournament as rrt


class TestParseResult(unittest.TestCase):
    def test_basic_parse(self):
        self.assertEqual(rrt.parse_result("WINS=5 LOSSES=3 DRAWS=2"), (5, 3, 2))

    def test_with_surrounding_text(self):
        stdout = "some output\nWINS=10 LOSSES=0 DRAWS=5\nmore text"
        self.assertEqual(rrt.parse_result(stdout), (10, 0, 5))

    def test_no_match(self):
        self.assertIsNone(rrt.parse_result("no results here"))

    def test_empty_string(self):
        self.assertIsNone(rrt.parse_result(""))

    def test_zeros(self):
        self.assertEqual(rrt.parse_result("WINS=0 LOSSES=0 DRAWS=0"), (0, 0, 0))

    def test_large_numbers(self):
        self.assertEqual(rrt.parse_result("WINS=999 LOSSES=888 DRAWS=777"), (999, 888, 777))


class TestBuildCmd(unittest.TestCase):
    def test_tiered_vs_tiered(self):
        cmd = rrt.build_cmd("/a.pt", "/b.pt", "tiered", "tiered", 10, 200, 128, 28, 1.0, 0)
        self.assertIn("--enable-koth", cmd)
        self.assertNotIn("--candidate-disable-tier1", cmd)
        self.assertNotIn("--current-disable-tier1", cmd)

    def test_vanilla_vs_tiered(self):
        cmd = rrt.build_cmd("/a.pt", "/b.pt", "vanilla", "tiered", 10, 200, 128, 28, 1.0, 0)
        self.assertIn("--candidate-disable-tier1", cmd)
        self.assertIn("--candidate-disable-material", cmd)
        self.assertNotIn("--current-disable-tier1", cmd)

    def test_tiered_vs_vanilla(self):
        cmd = rrt.build_cmd("/a.pt", "/b.pt", "tiered", "vanilla", 10, 200, 128, 28, 1.0, 0)
        self.assertNotIn("--candidate-disable-tier1", cmd)
        self.assertIn("--current-disable-tier1", cmd)
        self.assertIn("--current-disable-material", cmd)

    def test_vanilla_vs_vanilla(self):
        cmd = rrt.build_cmd("/a.pt", "/b.pt", "vanilla", "vanilla", 10, 200, 128, 28, 1.0, 0)
        self.assertIn("--candidate-disable-tier1", cmd)
        self.assertIn("--current-disable-tier1", cmd)

    def test_paths_and_args(self):
        cmd = rrt.build_cmd("/cand.pt", "/curr.pt", "tiered", "tiered", 20, 400, 64, 8, 2.5, 42)
        self.assertEqual(cmd[1], "/cand.pt")
        self.assertEqual(cmd[2], "/curr.pt")
        self.assertEqual(cmd[3], "20")
        self.assertEqual(cmd[4], "400")
        idx = cmd.index("--batch-size")
        self.assertEqual(cmd[idx + 1], "64")
        idx = cmd.index("--threads")
        self.assertEqual(cmd[idx + 1], "8")
        idx = cmd.index("--explore-base")
        self.assertEqual(cmd[idx + 1], "2.5")
        idx = cmd.index("--seed-offset")
        self.assertEqual(cmd[idx + 1], "42")


class TestComputeEloMle(unittest.TestCase):
    def test_equal_results(self):
        """Equal win rates should give equal Elo."""
        results = {
            (0, 1): (10, 10, 0),
            (0, 2): (10, 10, 0),
            (1, 2): (10, 10, 0),
        }
        elos = rrt.compute_elo_mle(results, ["A", "B", "C"])
        # All Elos should be close to 0
        for e in elos:
            self.assertAlmostEqual(e, 0.0, places=1)

    def test_clear_ranking(self):
        """A > B > C should produce decreasing Elos."""
        results = {
            (0, 1): (30, 10, 0),
            (0, 2): (35, 5, 0),
            (1, 2): (25, 15, 0),
        }
        elos = rrt.compute_elo_mle(results, ["A", "B", "C"])
        self.assertGreater(elos[0], elos[1])
        self.assertGreater(elos[1], elos[2])

    def test_anchor_at_zero(self):
        """Anchor model should have Elo 0."""
        results = {(0, 1): (20, 10, 0)}
        elos = rrt.compute_elo_mle(results, ["A", "B"], anchor_idx=0)
        self.assertAlmostEqual(elos[0], 0.0)

    def test_draws_count(self):
        """Draws should split value equally."""
        # All draws = equal strength
        results = {(0, 1): (0, 0, 20)}
        elos = rrt.compute_elo_mle(results, ["A", "B"])
        self.assertAlmostEqual(elos[0], 0.0, places=1)
        self.assertAlmostEqual(elos[1], 0.0, places=1)

    def test_two_models_winner_positive(self):
        """Winner should get positive Elo (relative to anchor=loser)."""
        results = {(0, 1): (5, 15, 0)}
        elos = rrt.compute_elo_mle(results, ["A", "B"], anchor_idx=0)
        self.assertAlmostEqual(elos[0], 0.0)
        self.assertGreater(elos[1], 0.0)

    def test_missing_pair_ignored(self):
        """Models with no head-to-head get default Elo."""
        results = {(0, 1): (20, 10, 0)}
        elos = rrt.compute_elo_mle(results, ["A", "B", "C"])
        # C has no results, should stay near anchor
        self.assertAlmostEqual(elos[2], 0.0, places=1)

    def test_reversed_key_lookup(self):
        """Results keyed as (j, i) should still be found for model i."""
        results = {(1, 0): (10, 20, 0)}  # Model 1 wins=10, model 0 wins=20
        elos = rrt.compute_elo_mle(results, ["A", "B"], anchor_idx=0)
        # A (idx 0) won 20 out of 30, should be positive relative to B
        self.assertGreater(elos[0], elos[1])

    def test_perfect_score(self):
        """100% win rate should give high positive Elo."""
        results = {(0, 1): (20, 0, 0)}
        elos = rrt.compute_elo_mle(results, ["A", "B"], anchor_idx=1)
        self.assertGreater(elos[0], 200)


class TestSaveAndLoadResults(unittest.TestCase):
    def test_roundtrip(self):
        """Save and load should preserve results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_names = ["A", "B", "C"]
            elos = [100.0, 0.0, -50.0]
            results = {(0, 1): (15, 5, 10), (0, 2): (20, 3, 7), (1, 2): (12, 8, 10)}
            games_played = {(0, 1): 30, (0, 2): 30, (1, 2): 30}

            rrt.save_results(model_names, elos, results, games_played, tmpdir)
            loaded_results, loaded_gp = rrt.load_partial_results(tmpdir)

            self.assertEqual(loaded_results, results)
            self.assertEqual(loaded_gp, games_played)

    def test_load_nonexistent(self):
        """Loading from nonexistent dir returns empty dicts."""
        results, gp = rrt.load_partial_results("/nonexistent/path")
        self.assertEqual(results, {})
        self.assertEqual(gp, {})

    def test_save_creates_dir(self):
        """Save should create the output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "new_subdir")
            rrt.save_results(["A"], [0.0], {}, {}, subdir)
            self.assertTrue(os.path.exists(subdir))

    def test_json_format(self):
        """Saved JSON should have expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rrt.save_results(["A", "B"], [10.0, -10.0],
                             {(0, 1): (5, 3, 2)}, {(0, 1): 10}, tmpdir)
            with open(os.path.join(tmpdir, "round_robin_results.json")) as f:
                data = json.load(f)
            self.assertIn("models", data)
            self.assertIn("elos", data)
            self.assertIn("results", data)
            self.assertIn("games_played", data)
            self.assertIn("timestamp", data)
            self.assertEqual(data["models"], ["A", "B"])
            self.assertEqual(data["results"]["0,1"], [5, 3, 2])


class TestBootstrapConsecutiveCi(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_returns_n_minus_1_gaps(self):
        """Should return one gap per consecutive pair."""
        results = {
            (0, 1): (20, 10, 0),
            (0, 2): (25, 5, 0),
            (1, 2): (15, 10, 5),
        }
        gaps = rrt.bootstrap_consecutive_ci(results, ["A", "B", "C"], n_bootstrap=50)
        self.assertEqual(len(gaps), 2)  # 3 models -> 2 consecutive gaps

    def test_empty_results(self):
        """Empty results should return empty list."""
        gaps = rrt.bootstrap_consecutive_ci({}, ["A", "B", "C"])
        self.assertEqual(gaps, [])

    def test_sorted_by_ci_descending(self):
        """Gaps should be sorted by CI width descending."""
        results = {
            (0, 1): (15, 5, 10),
            (0, 2): (20, 3, 7),
            (0, 3): (25, 2, 3),
            (1, 2): (12, 8, 10),
            (1, 3): (18, 5, 7),
            (2, 3): (14, 6, 10),
        }
        gaps = rrt.bootstrap_consecutive_ci(results, ["A", "B", "C", "D"], n_bootstrap=100)
        ci_widths = [g[3] for g in gaps]
        self.assertEqual(ci_widths, sorted(ci_widths, reverse=True))

    def test_gap_structure(self):
        """Each gap should be (higher_idx, lower_idx, mean, ci, games)."""
        results = {(0, 1): (20, 10, 0)}
        gaps = rrt.bootstrap_consecutive_ci(results, ["A", "B"], n_bootstrap=50)
        self.assertEqual(len(gaps), 1)
        higher, lower, mean, ci, gp = gaps[0]
        self.assertIsInstance(higher, int)
        self.assertIsInstance(lower, int)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(ci, float)
        self.assertGreater(ci, 0)
        self.assertEqual(gp, 30)  # 20 + 10 + 0

    def test_ci_positive(self):
        """CI widths should be positive."""
        results = {
            (0, 1): (15, 10, 5),
            (1, 2): (12, 8, 10),
            (0, 2): (18, 7, 5),
        }
        gaps = rrt.bootstrap_consecutive_ci(results, ["A", "B", "C"], n_bootstrap=100)
        for _, _, _, ci, _ in gaps:
            self.assertGreater(ci, 0)

    def test_more_games_reduces_ci(self):
        """More games should generally reduce CI width."""
        random.seed(42)
        small_results = {(0, 1): (6, 4, 0)}
        gaps_small = rrt.bootstrap_consecutive_ci(small_results, ["A", "B"], n_bootstrap=200)

        random.seed(42)
        large_results = {(0, 1): (60, 40, 0)}
        gaps_large = rrt.bootstrap_consecutive_ci(large_results, ["A", "B"], n_bootstrap=200)

        # Larger sample -> smaller CI (with same win rate)
        self.assertGreater(gaps_small[0][3], gaps_large[0][3])

    def test_games_played_count(self):
        """Games played should reflect total games in that pair."""
        results = {(0, 1): (10, 5, 5)}
        gaps = rrt.bootstrap_consecutive_ci(results, ["A", "B"], n_bootstrap=10)
        self.assertEqual(gaps[0][4], 20)  # 10+5+5

    def test_games_count_reversed_key(self):
        """Should find games even when pair key is reversed from ranking order."""
        # If model 1 is ranked higher but pair is keyed (0, 1)
        results = {(0, 1): (5, 20, 5)}  # model 1 wins more
        gaps = rrt.bootstrap_consecutive_ci(results, ["A", "B"], n_bootstrap=10)
        self.assertEqual(gaps[0][4], 30)

    def test_higher_ranked_first(self):
        """The higher-ranked model index should come first in gap tuple."""
        results = {
            (0, 1): (20, 10, 0),
            (0, 2): (25, 5, 0),
            (1, 2): (15, 10, 5),
        }
        elos = rrt.compute_elo_mle(results, ["A", "B", "C"])
        ranked = sorted(range(3), key=lambda i: elos[i], reverse=True)

        gaps = rrt.bootstrap_consecutive_ci(results, ["A", "B", "C"], n_bootstrap=50)
        for higher, lower, _, _, _ in gaps:
            self.assertIn(higher, ranked)
            self.assertIn(lower, ranked)
            self.assertGreater(elos[higher], elos[lower])


class TestPrintCiTable(unittest.TestCase):
    def test_output_contains_model_names(self):
        """CI table should contain model names."""
        gaps = [(0, 1, 50.0, 30.0, 20)]
        model_names = ["Alpha", "Beta"]
        elos = [50.0, 0.0]

        captured = StringIO()
        sys.stdout = captured
        rrt.print_ci_table(gaps, model_names, elos)
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("Alpha", output)
        self.assertIn("Beta", output)

    def test_focus_pair_marker(self):
        """Focus pair should have NEXT marker."""
        gaps = [(0, 1, 50.0, 30.0, 20), (1, 2, 20.0, 15.0, 20)]
        model_names = ["A", "B", "C"]
        elos = [100.0, 50.0, 0.0]

        captured = StringIO()
        sys.stdout = captured
        rrt.print_ci_table(gaps, model_names, elos, focus_pair=(0, 1))
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("<-- NEXT", output)

    def test_no_focus_pair(self):
        """Without focus pair, no NEXT marker."""
        gaps = [(0, 1, 50.0, 30.0, 20)]
        model_names = ["A", "B"]
        elos = [50.0, 0.0]

        captured = StringIO()
        sys.stdout = captured
        rrt.print_ci_table(gaps, model_names, elos)
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertNotIn("<-- NEXT", output)


class TestPrintRatings(unittest.TestCase):
    def test_output_sorted_by_elo(self):
        """Models should be printed in Elo order."""
        model_names = ["Weak", "Strong", "Mid"]
        elos = [-100.0, 100.0, 0.0]
        results = {(0, 1): (5, 15, 0), (0, 2): (8, 12, 0), (1, 2): (14, 6, 0)}

        captured = StringIO()
        sys.stdout = captured
        rrt.print_ratings(model_names, elos, results, {})
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        strong_pos = output.find("Strong")
        mid_pos = output.find("Mid")
        weak_pos = output.find("Weak")
        self.assertLess(strong_pos, mid_pos)
        self.assertLess(mid_pos, weak_pos)

    def test_win_rate_calculation(self):
        """Win rate should be (W + 0.5*D) / total."""
        model_names = ["A", "B"]
        elos = [50.0, -50.0]
        results = {(0, 1): (6, 2, 2)}  # WR = (6 + 1) / 10 = 0.700

        captured = StringIO()
        sys.stdout = captured
        rrt.print_ratings(model_names, elos, results, {})
        sys.stdout = sys.__stdout__

        self.assertIn("0.700", captured.getvalue())


class TestPrintCrossTable(unittest.TestCase):
    def test_contains_cross_table_header(self):
        """Should print CROSS-TABLE header."""
        captured = StringIO()
        sys.stdout = captured
        rrt.print_cross_table(["A", "B"], [10.0, -10.0], {(0, 1): (7, 3, 0)})
        sys.stdout = sys.__stdout__

        self.assertIn("CROSS-TABLE", captured.getvalue())

    def test_diagonal_dashes(self):
        """Diagonal should show ---."""
        captured = StringIO()
        sys.stdout = captured
        rrt.print_cross_table(["A", "B"], [10.0, -10.0], {(0, 1): (7, 3, 0)})
        sys.stdout = sys.__stdout__

        self.assertIn("---", captured.getvalue())

    def test_missing_pair_shows_question(self):
        """Missing pair should show ?."""
        captured = StringIO()
        sys.stdout = captured
        rrt.print_cross_table(["A", "B", "C"], [10.0, 0.0, -10.0], {(0, 1): (7, 3, 0)})
        sys.stdout = sys.__stdout__

        self.assertIn("?", captured.getvalue())


class TestRunAdaptivePhase(unittest.TestCase):
    def _make_args(self, **kwargs):
        defaults = {
            'batch': 10, 'sims': 200, 'batch_size': 128, 'threads': 28,
            'explore_base': 1.0, 'max_total_games': 9000, 'ci_target': 50.0,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_stops_on_max_total_games(self):
        """Should stop when total games >= max_total_games."""
        results = {(0, 1): (50, 50, 0)}  # 100 total games
        games_played = {(0, 1): 100}
        args = self._make_args(max_total_games=100)

        captured = StringIO()
        sys.stdout = captured
        rrt.run_adaptive_phase(["A", "B"], results, games_played, args, "/tmp", 0)
        sys.stdout = sys.__stdout__

        self.assertIn("max total games", captured.getvalue())

    def test_stops_on_ci_target(self):
        """Should stop when max CI < ci_target."""
        random.seed(42)
        # Large number of games -> small CI
        results = {(0, 1): (300, 200, 100)}
        games_played = {(0, 1): 600}
        args = self._make_args(ci_target=999.0, max_total_games=99999)

        captured = StringIO()
        sys.stdout = captured
        rrt.run_adaptive_phase(["A", "B"], results, games_played, args, "/tmp", 0)
        sys.stdout = sys.__stdout__

        self.assertIn("below target", captured.getvalue())

    def test_stops_on_empty_results(self):
        """Should stop when no results to analyze."""
        args = self._make_args()

        captured = StringIO()
        sys.stdout = captured
        rrt.run_adaptive_phase(["A", "B"], {}, {}, args, "/tmp", 0)
        sys.stdout = sys.__stdout__

        # Should hit either empty gaps or max_total_games=0
        output = captured.getvalue()
        self.assertTrue("No consecutive gaps" in output or "max total games" in output)

    @patch('round_robin_tournament.run_batch')
    @patch('round_robin_tournament.save_results')
    def test_plays_most_uncertain_pair(self, mock_save, mock_run_batch):
        """Should play games for the pair with widest CI."""
        random.seed(42)
        mock_run_batch.return_value = (5, 3, 2)

        models = [("A", "/a.pt", "tiered"), ("B", "/b.pt", "tiered"),
                  ("C", "/c.pt", "tiered"), ("D", "/d.pt", "tiered")]

        # 4 models, pair (0,1) has fewer games -> more uncertain
        results = {
            (0, 1): (5, 5, 0),    # 10 games, uncertain
            (0, 2): (50, 30, 20), # 100 games, more certain
            (0, 3): (50, 30, 20),
            (1, 2): (50, 30, 20),
            (1, 3): (50, 30, 20),
            (2, 3): (50, 30, 20),
        }
        games_played = {
            (0, 1): 10,
            (0, 2): 100, (0, 3): 100,
            (1, 2): 100, (1, 3): 100, (2, 3): 100,
        }
        # Only run one iteration
        args = self._make_args(max_total_games=520, batch=10, ci_target=1.0)

        captured = StringIO()
        sys.stdout = captured
        with tempfile.TemporaryDirectory() as tmpdir:
            rrt.run_adaptive_phase(["A", "B", "C", "D"], results, games_played,
                                   args, tmpdir, 0, models=models)
        sys.stdout = sys.__stdout__

        # Should have called run_batch at least once
        self.assertTrue(mock_run_batch.called)

    @patch('round_robin_tournament.run_batch')
    @patch('round_robin_tournament.save_results')
    def test_accumulates_results(self, mock_save, mock_run_batch):
        """Results from adaptive games should accumulate."""
        random.seed(42)
        mock_run_batch.return_value = (7, 2, 1)

        models = [("A", "/a.pt", "tiered"), ("B", "/b.pt", "tiered")]
        results = {(0, 1): (10, 10, 0)}
        games_played = {(0, 1): 20}
        args = self._make_args(max_total_games=35, batch=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            captured = StringIO()
            sys.stdout = captured
            rrt.run_adaptive_phase(["A", "B"], results, games_played, args, tmpdir, 0,
                                   models=models)
            sys.stdout = sys.__stdout__

        # Results should have been updated
        w, l, d = results[(0, 1)]
        self.assertGreater(w + l + d, 20)

    @patch('round_robin_tournament.run_batch')
    @patch('round_robin_tournament.save_results')
    def test_failed_batch_continues(self, mock_save, mock_run_batch):
        """A failed batch should not crash the adaptive phase."""
        random.seed(42)
        # First call fails, second would succeed but we hit game limit
        mock_run_batch.side_effect = [None, (5, 3, 2)]

        models = [("A", "/a.pt", "tiered"), ("B", "/b.pt", "tiered")]
        results = {(0, 1): (10, 10, 0)}
        games_played = {(0, 1): 20}
        # After failed batch, total stays at 20. Next iteration plays 5 -> 25 -> stops
        args = self._make_args(max_total_games=25, batch=5)

        captured = StringIO()
        sys.stdout = captured
        with tempfile.TemporaryDirectory() as tmpdir:
            rrt.run_adaptive_phase(["A", "B"], results, games_played, args, tmpdir, 0,
                                   models=models)
        sys.stdout = sys.__stdout__

        self.assertIn("FAILED", captured.getvalue())


class TestRunBatch(unittest.TestCase):
    @patch('round_robin_tournament.subprocess.run')
    def test_returns_parsed_result(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(
            stdout="WINS=8 LOSSES=1 DRAWS=1",
            stderr="",
        )
        result = rrt.run_batch("A", "/a.pt", "tiered", "B", "/b.pt", "tiered",
                               10, 200, 128, 28, 1.0, 0)
        self.assertEqual(result, (8, 1, 1))

    @patch('round_robin_tournament.subprocess.run')
    def test_returns_none_on_parse_failure(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(
            stdout="garbage output",
            stderr="",
        )
        result = rrt.run_batch("A", "/a.pt", "tiered", "B", "/b.pt", "tiered",
                               10, 200, 128, 28, 1.0, 0)
        self.assertIsNone(result)

    @patch('round_robin_tournament.subprocess.run')
    def test_prints_stderr_summary(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(
            stdout="WINS=5 LOSSES=5 DRAWS=0",
            stderr="line1\nline2\nline3\nline4\nline5",
        )
        captured = StringIO()
        sys.stdout = captured
        rrt.run_batch("A", "/a.pt", "tiered", "B", "/b.pt", "tiered",
                      10, 200, 128, 28, 1.0, 0)
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        # Should print last 2 lines when > 3 lines of stderr
        self.assertIn("line4", output)
        self.assertIn("line5", output)


class TestBootstrapDeterminism(unittest.TestCase):
    def test_same_seed_same_results(self):
        """Bootstrap should be deterministic with same random seed."""
        results = {
            (0, 1): (15, 10, 5),
            (0, 2): (20, 5, 5),
            (1, 2): (12, 8, 10),
        }
        model_names = ["A", "B", "C"]

        random.seed(123)
        gaps1 = rrt.bootstrap_consecutive_ci(results, model_names, n_bootstrap=100)

        random.seed(123)
        gaps2 = rrt.bootstrap_consecutive_ci(results, model_names, n_bootstrap=100)

        for g1, g2 in zip(gaps1, gaps2):
            self.assertEqual(g1[0], g2[0])
            self.assertEqual(g1[1], g2[1])
            self.assertAlmostEqual(g1[2], g2[2])
            self.assertAlmostEqual(g1[3], g2[3])
            self.assertEqual(g1[4], g2[4])


class TestEdgeCases(unittest.TestCase):
    def test_two_models_bootstrap(self):
        """Bootstrap should work with just 2 models."""
        random.seed(42)
        results = {(0, 1): (15, 5, 10)}
        gaps = rrt.bootstrap_consecutive_ci(results, ["A", "B"], n_bootstrap=50)
        self.assertEqual(len(gaps), 1)

    def test_all_draws_elo(self):
        """All draws should give equal Elo."""
        results = {(0, 1): (0, 0, 50)}
        elos = rrt.compute_elo_mle(results, ["A", "B"])
        self.assertAlmostEqual(elos[0], elos[1], places=1)

    def test_bootstrap_with_all_zero_games_in_pair(self):
        """Pairs with 0 total games should be skipped."""
        results = {(0, 1): (0, 0, 0), (0, 2): (10, 5, 5)}
        gaps = rrt.bootstrap_consecutive_ci(results, ["A", "B", "C"], n_bootstrap=50)
        # Should still return gaps (using pairs that have data)
        self.assertGreater(len(gaps), 0)

    def test_compute_elo_all_zero_games(self):
        """All-zero results should not crash."""
        results = {(0, 1): (0, 0, 0)}
        elos = rrt.compute_elo_mle(results, ["A", "B"])
        self.assertEqual(len(elos), 2)

    def test_four_model_bootstrap_gap_count(self):
        """4 models should produce 3 consecutive gaps."""
        random.seed(42)
        results = {
            (0, 1): (15, 5, 10),
            (0, 2): (20, 3, 7),
            (0, 3): (25, 2, 3),
            (1, 2): (12, 8, 10),
            (1, 3): (18, 5, 7),
            (2, 3): (14, 6, 10),
        }
        gaps = rrt.bootstrap_consecutive_ci(results, ["A", "B", "C", "D"], n_bootstrap=50)
        self.assertEqual(len(gaps), 3)


class TestSaveLoadRoundtrip(unittest.TestCase):
    def test_multiple_saves_overwrite(self):
        """Later saves should overwrite earlier ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rrt.save_results(["A", "B"], [10.0, -10.0],
                             {(0, 1): (5, 3, 2)}, {(0, 1): 10}, tmpdir)
            rrt.save_results(["A", "B"], [20.0, -20.0],
                             {(0, 1): (15, 5, 10)}, {(0, 1): 30}, tmpdir)

            results, gp = rrt.load_partial_results(tmpdir)
            self.assertEqual(results[(0, 1)], (15, 5, 10))
            self.assertEqual(gp[(0, 1)], 30)

    def test_preserves_multiple_pairs(self):
        """Should preserve results for all pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {(0, 1): (10, 5, 5), (0, 2): (8, 7, 5), (1, 2): (12, 3, 5)}
            gp = {(0, 1): 20, (0, 2): 20, (1, 2): 20}
            rrt.save_results(["A", "B", "C"], [50, 0, -50], results, gp, tmpdir)

            loaded_r, loaded_gp = rrt.load_partial_results(tmpdir)
            self.assertEqual(len(loaded_r), 3)
            self.assertEqual(loaded_r[(0, 2)], (8, 7, 5))


if __name__ == "__main__":
    unittest.main()

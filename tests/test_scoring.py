import unittest

import numpy as np

from badminton_analysis.analysis.evaluate_scoring import evaluate
from badminton_analysis.analysis.line_calls import RallyCall
from badminton_analysis.analysis.scoreboard import detect_score_changes
from badminton_analysis.analysis.scoring import MatchScore, rally_winner_side, score_rallies


def call(rally, end_sec, result, side, flags=None, confidence="high"):
    return RallyCall(rally=rally, start_frame=0, end_frame=0, start_sec=end_sec - 5, end_sec=end_sec,
                     landing_side=side, call=result, confidence=confidence, flags=flags or [])


class MatchScoreTests(unittest.TestCase):
    def test_game_to_21(self):
        match = MatchScore()
        for _ in range(20):
            match.award(0)
        event = match.award(0)
        self.assertTrue(event["game_over"])
        self.assertEqual(match.games, [1, 0])
        self.assertEqual(match.points, [0, 0])

    def test_win_by_two_and_cap_at_30(self):
        match = MatchScore(start_points=(20, 20))
        self.assertFalse(match.award(0)["game_over"])   # 21-20
        self.assertFalse(match.award(1)["game_over"])   # 21-21
        match = MatchScore(start_points=(29, 29))
        self.assertTrue(match.award(1)["game_over"])    # 29-30
        self.assertEqual(match.games, [0, 1])

    def test_ends_change_after_game_and_at_11_in_deciding_game(self):
        match = MatchScore(player0_side="lower")
        self.assertEqual(match.side_of(0), "lower")
        for _ in range(21):
            match.award(0)
        self.assertEqual(match.side_of(0), "upper")
        for _ in range(21):
            match.award(1)
        self.assertEqual(match.side_of(0), "lower")     # game 3 starts on game 1 ends
        for _ in range(10):
            match.award(0)
        self.assertFalse(match.award(1)["change_ends"])
        self.assertTrue(match.award(0)["change_ends"])  # 11-1
        self.assertEqual(match.side_of(0), "upper")

    def test_match_over_after_two_games(self):
        match = MatchScore(start_games=(1, 0), start_points=(20, 5))
        event = match.award(0)
        self.assertTrue(event["match_over"])
        self.assertTrue(match.match_over)

    def test_rally_winner_serves_from_court_by_score(self):
        match = MatchScore(start_points=(3, 4))
        match.award(1)
        self.assertEqual(match.server, 1)
        self.assertEqual(match.service_court(), "left")  # 5 is odd


class RallyWinnerTests(unittest.TestCase):
    def test_in_wins_for_the_other_side(self):
        self.assertEqual(rally_winner_side(call(1, 10, "in", "upper"))[0], "lower")

    def test_out_loses_for_the_hitter_across_the_net(self):
        self.assertEqual(rally_winner_side(call(1, 10, "out", "upper"))[0], "upper")

    def test_score_rallies_maps_sides_to_players(self):
        calls = [call(1, 10, "in", "upper"), call(2, 20, "out", "upper")]
        results, match = score_rallies(calls, player0_side="lower")
        self.assertEqual([r.winner for r in results], [0, 1])
        self.assertEqual(match.points, [1, 1])

    def test_landing_beside_net_lowers_confidence(self):
        calls = [call(1, 10, "in", "upper", flags=["landed 20 cm from the net: which side it fell on is uncertain"])]
        results, _ = score_rallies(calls)
        self.assertEqual(results[0].confidence, "low")


class ScoreboardTests(unittest.TestCase):
    def frames(self, states, repeat=4):
        t = 0.0
        for state in states:
            for _ in range(repeat):
                yield t, [np.full((10, 10), state[0], np.uint8), np.full((10, 10), state[1], np.uint8)]
                t += 0.2

    def test_one_box_changing_is_a_point_to_that_row(self):
        changes = detect_score_changes(self.frames([(50, 50), (120, 50), (120, 200)]))
        self.assertEqual([(c.rows, c.kind) for c in changes], [([0], "point"), ([1], "point")])

    def test_brief_flicker_is_ignored(self):
        samples = list(self.frames([(50, 50)]))
        samples.append((1.0, [np.full((10, 10), 200, np.uint8), np.full((10, 10), 50, np.uint8)]))
        samples += [(1.2 + i * 0.2, crops) for i, (_, crops) in enumerate(self.frames([(50, 50)]))]
        self.assertEqual(detect_score_changes(samples), [])

    def test_both_boxes_changing_is_not_a_point(self):
        changes = detect_score_changes(self.frames([(50, 50), (200, 200)]))
        self.assertEqual([c.kind for c in changes], ["both"])


class EvaluateTests(unittest.TestCase):
    def test_counts_correct_wrong_missed_and_extra(self):
        # Player 1 (row 0) starts on the near (lower) side.
        calls = [
            call(1, 10, "in", "upper"),    # near wins -> row 0, true row 0: correct
            call(2, 18, "in", "upper"),    # replay or false rally before the real one: extra
            call(3, 20, "in", "upper"),    # near wins -> row 0, true row 1: wrong
        ]
        truth = [(12.0, 0), (23.0, 1), (40.0, 0)]  # third point has no rally: missed
        report = evaluate(calls, truth, player0_side="lower")
        self.assertEqual((report["correct"], report["wrong"], report["missed"]), (1, 1, 1))
        self.assertEqual(report["extra_rallies"], 1)

    def test_side_mapping_follows_true_ends(self):
        # Row 0 wins game 1 21-0 on the near side, then is on the far side in game 2.
        calls = [call(1, 1000, "in", "lower")]  # far side wins
        truth = [(float(i * 10), 0) for i in range(21)] + [(1001.0, 0)]
        report = evaluate(calls, truth, player0_side="lower")
        self.assertEqual(report["points"][-1]["result"], "correct")


if __name__ == "__main__":
    unittest.main()

import csv
import os
import tempfile
import unittest

from badminton_analysis.analysis.line_calls import RallyCall
from badminton_analysis.analysis.shuttleset import evaluate_shuttleset, load_truth, write_court_annotation


MATCH = "Player_A_Player_B_Test_Open_Final"
# Court corners from the real Marin v Yigit labels (1280x720 pixels).
CORNERS = {"upleft_x": 411.2, "upright_x": 857.8, "downleft_x": 295.8, "downright_x": 983.0,
           "upleft_y": 309.8, "upright_y": 311.4, "downleft_y": 658.4, "downright_y": 659.4}
SHOT_FIELDS = ["rally", "ball_round", "time", "frame_num", "roundscore_A", "roundscore_B", "player",
               "landing_x", "landing_y", "lose_reason", "getpoint_player", "player_location_y"]


def shot(rally, frame, player, location_y, score=(0, 0), landing=("", ""), lose_reason="", getpoint=""):
    return {"rally": rally, "ball_round": 1, "time": "00:00:00", "frame_num": frame,
            "roundscore_A": score[0], "roundscore_B": score[1], "player": player,
            "landing_x": landing[0], "landing_y": landing[1], "lose_reason": lose_reason,
            "getpoint_player": getpoint, "player_location_y": location_y}


class ShuttleSetTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        with open(os.path.join(self.dir, "match.csv"), "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, ["video", "winner", "loser"])
            writer.writeheader()
            writer.writerow({"video": MATCH, "winner": "Player A", "loser": "Player B"})
        with open(os.path.join(self.dir, "homography.csv"), "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, ["video", *CORNERS])
            writer.writeheader()
            writer.writerow({"video": MATCH, **CORNERS})
        os.makedirs(os.path.join(self.dir, MATCH))
        # A plays from the near side (y 550 is below the net).
        shots = [
            shot(1, 300, "A", 550),
            # B, on the far side, hits long past the near baseline: out, point to A.
            shot(1, 330, "B", 350, (1, 0), (640, 700), "出界", "A"),
            shot(2, 900, "A", 550),
            # A's shot lands in on the far side: point to A.
            shot(2, 930, "A", 550, (2, 0), (640, 380), "對手落地致勝", "A"),
            # A hits the net: point to B.
            shot(3, 1500, "A", 550, (2, 1), (630, 470), "掛網", "B"),
        ]
        with open(os.path.join(self.dir, MATCH, "set1.csv"), "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, SHOT_FIELDS)
            writer.writeheader()
            writer.writerows(shots)

    def test_load_truth_reads_winners_sides_reasons_and_landings(self):
        truths, info, _ = load_truth(self.dir, MATCH, fps=30.0)
        self.assertEqual(info["winner"], "Player A")
        self.assertEqual([t.reason for t in truths], ["out", "in", "net"])
        self.assertEqual([t.winner_side for t in truths], ["lower", "lower", "upper"])
        self.assertTrue(all(t.a_side == "lower" for t in truths))
        self.assertEqual(truths[0].start_sec, 10.0)
        self.assertEqual(truths[0].end_sec, 11.0)
        self.assertGreater(truths[0].landing_court[1], 13.4)   # past the near baseline
        self.assertLess(truths[1].landing_court[1], 6.7)       # far half

    def test_labels_scale_to_the_video_size(self):
        truths, _, corners = load_truth(self.dir, MATCH, video_size=(1920, 1080))
        self.assertAlmostEqual(corners[0][0], 411.2 * 1.5, places=3)
        self.assertEqual(truths[0].landing_image, [960.0, 1050.0])

    def test_write_court_annotation(self):
        out = os.path.join(self.dir, "out")
        path, corners = write_court_annotation(self.dir, MATCH, out, (1280, 720))
        self.assertEqual(corners[0], (411, 310))
        with open(path, encoding="utf-8") as file:
            self.assertTrue(file.readline().startswith("corners=[(411, 310)"))

    def test_evaluate_counts_winners_calls_and_scores(self):
        truths, _, _ = load_truth(self.dir, MATCH, fps=30.0)
        calls = [
            RallyCall(rally=1, start_frame=0, end_frame=0, start_sec=9, end_sec=11.9,
                      landing_side="lower", landing_court=[3.0, 13.8], call="out", confidence="high"),
            RallyCall(rally=2, start_frame=0, end_frame=0, start_sec=29, end_sec=31.5,
                      landing_side="upper", landing_court=[3.0, 4.0], call="in", confidence="high"),
            # Net rally: shuttle fell on A's own (near) side, so the far side wins. Correct.
            RallyCall(rally=3, start_frame=0, end_frame=0, start_sec=49, end_sec=50.5,
                      landing_side="lower", landing_court=[3.0, 7.0], call="in", confidence="high"),
            # A replay between rallies: extra.
            RallyCall(rally=4, start_frame=0, end_frame=0, start_sec=60, end_sec=70,
                      landing_side="lower", landing_court=[3.0, 10.0], call="in", confidence="low"),
        ]
        report = evaluate_shuttleset(calls, truths)
        self.assertEqual(report["rallies_found"], 3)
        self.assertEqual(report["extra_rallies"], 1)
        self.assertEqual(report["winner_correct"], 3)
        self.assertEqual((report["line_calls_correct"], report["line_calls_checked"]), (2, 2))
        self.assertEqual(report["predicted_set_scores"]["1"], [2, 1])
        self.assertEqual(report["true_set_scores"]["1"], [2, 1])

    def test_missing_rally_counts_as_missed(self):
        truths, _, _ = load_truth(self.dir, MATCH, fps=30.0)
        report = evaluate_shuttleset([], truths)
        self.assertEqual(report["winner_missed"], 3)
        self.assertEqual(report["rallies_found"], 0)


if __name__ == "__main__":
    unittest.main()

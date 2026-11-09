import json
import os
import tempfile
import unittest
from dataclasses import asdict

from badminton_analysis.analysis.line_calls import LineCallConfig, RallyCall, call_landing
from badminton_analysis.analysis.multi_camera import (COMBINED_CALLS, combine, find_offset, find_turn, load_camera,
                                                      turn)

AROUND = (True, True)


def rally_call(number, landing_frame, point, error, fps=30, method="settled"):
    verdict, margin = call_landing(point, "singles")
    return RallyCall(rally=number, start_frame=landing_frame - 5 * fps, end_frame=landing_frame,
                     start_sec=landing_frame / fps - 5, end_sec=landing_frame / fps, landing_frame=landing_frame,
                     landing_detect_frame=landing_frame, landing_image=[100.0, 200.0], landing_court=list(point),
                     landing_side="upper" if point[1] < 6.7 else "lower", call=verdict, margin_m=margin,
                     landing_error_m=error, landing_method=method, confidence="high")


class AlignmentTests(unittest.TestCase):
    def test_offset_lines_up_rallies_although_one_phone_missed_one(self):
        times_a = [12.0, 40.5, 71.2, 95.0, 130.4]
        times_b = [t - 7.25 for t in times_a if t != 71.2] + [52.0]   # missed one, saw a rally that was not
        offset, pairs = find_offset(times_a, times_b)
        self.assertAlmostEqual(offset, 7.25, places=6)
        self.assertEqual(len(pairs), 4)

    def test_offset_stays_near_the_start_times(self):
        offset, pairs = find_offset([10.0, 30.0], [10.0, 30.0], guess=40.0)
        self.assertEqual((offset, pairs), (40.0, []))

    def test_court_marked_from_the_other_end_is_turned_around(self):
        points = [[1.0, 1.0], [5.0, 12.0], [3.0, 6.0], [0.5, 10.0]]
        self.assertEqual(find_turn(points, [turn(point, AROUND) for point in points]), AROUND)
        self.assertEqual(find_turn(points, points), (False, False))


class CombineTests(unittest.TestCase):
    """Phone a behind one baseline, phone b behind the other, which marked its court from its own end, started
    3.2 s after a and films at 60 fps."""

    def setUp(self):
        self.config = LineCallConfig(match_type="singles")
        # true landings in a's court: one just inside a's far baseline, two near a
        landings = [(600, [2.0, 0.08]), (1500, [4.0, 12.0]), (2400, [1.5, 9.0])]
        self.a = [rally_call(1, 600, [2.0, -0.10], [0.05, 0.40]),   # far from a: 10 cm out, give or take 40
                  rally_call(2, 1500, [4.0, 12.02], [0.03, 0.04]),
                  rally_call(3, 2400, [1.5, 9.03], [0.03, 0.06])]
        self.b = [rally_call(number, round((frame / 30 - 3.2) * 60), turn(point, AROUND), error, fps=60)
                  for number, ((frame, point), error) in
                  enumerate(zip(landings, [[0.05, 0.05], [0.05, 0.35], [0.05, 0.25]]), 1)]

    def cameras(self, b_calls=None):
        return [{"name": "a", "calls": self.a, "fps": 30, "started": 1000.0},
                {"name": "b", "calls": self.b if b_calls is None else b_calls, "fps": 60, "started": 1003.2}]

    def test_the_phone_near_the_landing_decides_it(self):
        calls, alignment = combine(self.cameras(), self.config)
        self.assertEqual(alignment[1]["turn"], "turned around")
        self.assertEqual(alignment[1]["matched"], 3)
        self.assertAlmostEqual(alignment[1]["offset_s"], 3.2, places=2)
        first = calls[0]
        self.assertEqual(first.call, "in")             # a alone called it out
        self.assertAlmostEqual(first.margin_m, 0.08, delta=0.01)
        self.assertLessEqual(first.landing_error_m[1], 0.05)   # b's error alone: a's 0.40 hardly counts
        self.assertEqual(first.landing_method, "combined")
        self.assertEqual(first.landing_frame, 600)
        self.assertEqual([seen["call"] for seen in first.cameras], ["out", "in"])
        self.assertIn("phones split: a OUT, b IN", first.flags)
        self.assertEqual(first.confidence, "medium")
        self.assertEqual(calls[1].call, "in")
        self.assertEqual(calls[1].confidence, "high")

    def test_phones_that_disagree_leave_the_call_for_review(self):
        self.b[2] = rally_call(3, self.b[2].landing_frame, turn([1.5, 4.0], AROUND), [0.05, 0.05], fps=60)
        calls, _ = combine(self.cameras(), self.config)
        self.assertEqual(calls[2].confidence, "low")
        self.assertTrue(any(flag.startswith("phones disagree") for flag in calls[2].flags))

    def test_a_rally_only_one_phone_saw_is_kept_in_the_first_phones_frames(self):
        extra = rally_call(4, round((3600 / 30 - 3.2) * 60), turn([3.0, 2.0], AROUND), [0.05, 0.05], fps=60)
        calls, _ = combine(self.cameras(self.b + [extra]), self.config)
        self.assertEqual(len(calls), 4)
        last = calls[3]
        self.assertEqual(last.landing_frame, 3600)
        self.assertIsNone(last.landing_image)
        self.assertEqual([seen["camera"] for seen in last.cameras], ["b"])
        self.assertEqual(last.landing_court, [3.0, 2.0])
        self.assertEqual(last.landing_side, "upper")

    def test_calls_are_read_from_an_analysis_and_a_recording(self):
        analysis, recording = tempfile.mkdtemp(), tempfile.mkdtemp()
        with open(os.path.join(analysis, "metadata.json"), "w", encoding="utf-8") as file:
            json.dump({"video": {"fps": 30}}, file)
        with open(os.path.join(analysis, "line_calls.json"), "w", encoding="utf-8") as file:
            json.dump({"config": {"match_type": "doubles"}, "rallies": [asdict(call) for call in self.a]}, file)
        with open(os.path.join(recording, "capture.json"), "w", encoding="utf-8") as file:
            json.dump({"fps": 60, "started": 1003.2}, file)
        with open(os.path.join(recording, "live_line_calls.json"), "w", encoding="utf-8") as file:
            json.dump([asdict(call) for call in self.b], file)
        a, b = load_camera(analysis), load_camera(recording)
        self.assertEqual((a["fps"], a["match_type"], a["started"], len(a["calls"])), (30, "doubles", None, 3))
        self.assertEqual((b["fps"], b["started"], b["calls"][0].landing_court), (60, 1003.2, self.b[0].landing_court))
        calls, alignment = combine([a, b], self.config)
        self.assertEqual(alignment[1]["matched"], 3)   # no start time for a: every shift is tried
        self.assertEqual(calls[0].call, "in")
        self.assertFalse(os.path.exists(os.path.join(analysis, COMBINED_CALLS)))


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest

import numpy as np

from badminton_analysis.analysis.line_calls import (
    TRACKNET_CSV,
    LineCallConfig,
    call_landing,
    compute_line_calls,
    load_run,
)
from badminton_analysis.court.mapper import CourtMapper
from badminton_analysis.court.reference import BADMINTON_SINGLES_MARGIN


FPS = 30
CORNERS = [(400, 250), (880, 250), (1000, 650), (280, 650)]
METADATA = {
    "video": {"fps": FPS, "height": 720, "width": 1280, "name": "synthetic", "path": "synthetic.mp4"},
    "court": {"corners": CORNERS},
    "outputs": {},
}


class SyntheticMatch:
    """Builds detections.jsonl-style records for scripted rallies."""

    def __init__(self):
        self.mapper = CourtMapper(CORNERS)
        self.records = []
        self.frame = 0

    def _image(self, court_point):
        return [float(v) for v in self.mapper.court_to_image(np.array(court_point, dtype=np.float32))]

    def _add(self, shuttle=None, hand=None, courts=None):
        self.frame += 1
        players = {"upper": None, "lower": None}
        for side, court in (courts or {}).items():
            players[side] = {"image": None, "court": list(court), "speed": None, "hands": {}}
        if hand is not None:
            players["lower"] = {"image": hand, "court": None, "speed": None,
                                "hands": {"left": None, "right": hand}}
        self.records.append({
            "frame": self.frame,
            "time_sec": self.frame / FPS,
            "detect_frame": self.frame,
            "players": players,
            "shuttlecock": {"image": shuttle},
        })

    def idle(self, seconds):
        for _ in range(int(seconds * FPS)):
            self._add()

    def rally(self, start_court, landing_court, flight_seconds=1.0, settle_seconds=1.0, held=False, skid_to=None,
              skid_frames=9, lost_from=None):
        start = np.array(self._image(start_court))
        end = np.array(self._image(landing_court))
        steps = int(flight_seconds * FPS)
        for i in range(steps + 1):
            u = i / steps
            if lost_from is not None and u > lost_from:  # lost on the way down: one repeated frame, then nothing
                self._add(shuttle=self.records[-1]["shuttlecock"]["image"] if (i - 1) / steps <= lost_from else None)
                continue
            point = start + (end - start) * u
            point[1] -= 220 * 4 * u * (1 - u)  # arc: the shuttle rises in the image mid-flight
            self._add(shuttle=[float(point[0]), float(point[1])])
        if skid_to is not None:  # slides on after touching down, then rests there
            rest = np.array(self._image(skid_to))
            for i in range(1, skid_frames + 1):
                point = end + (rest - end) * i / skid_frames
                self._add(shuttle=[float(point[0]), float(point[1])])
            end = rest
        hand = [float(end[0]) + 5, float(end[1]) - 5] if held else None
        for i in range(int(settle_seconds * FPS)):
            jitter = 1.0 if i % 2 else -1.0
            found = lost_from is None or i >= 10   # a lost shuttle is found again a little after it came to rest
            self._add(shuttle=[float(end[0]) + jitter, float(end[1])] if found else None, hand=hand)


class CallLandingTests(unittest.TestCase):
    def test_line_counts_as_in(self):
        call, margin = call_landing((BADMINTON_SINGLES_MARGIN, 3.0), "singles")
        self.assertEqual(call, "in")
        self.assertEqual(margin, 0.0)

    def test_doubles_alley_is_out_in_singles_only(self):
        self.assertEqual(call_landing((0.2, 3.0), "singles")[0], "out")
        self.assertEqual(call_landing((0.2, 3.0), "doubles")[0], "in")

    def test_long_is_out(self):
        call, margin = call_landing((3.0, 13.9), "doubles")
        self.assertEqual(call, "out")
        self.assertAlmostEqual(margin, -0.5, places=2)


class ComputeLineCallsTests(unittest.TestCase):
    def setUp(self):
        self.match = SyntheticMatch()
        self.match.idle(1)

    def calls(self, match_type="singles"):
        return compute_line_calls(METADATA, self.match.records, LineCallConfig(match_type=match_type))

    def test_splits_rallies_and_calls_each_landing(self):
        self.match.rally((3.0, 12.0), (3.0, 2.0))
        self.match.idle(3)
        self.match.rally((3.0, 2.0), (3.0, 13.9))
        self.match.idle(2)

        calls = self.calls()
        self.assertEqual(len(calls), 2)

        first, second = calls
        self.assertEqual((first.call, first.landing_side, first.confidence), ("in", "upper", "high"))
        self.assertEqual(first.landing_method, "settled")
        self.assertAlmostEqual(first.landing_court[1], 2.0, delta=0.05)

        self.assertEqual((second.call, second.landing_side), ("out", "lower"))
        self.assertAlmostEqual(second.margin_m, -0.5, delta=0.05)

    def test_match_type_changes_alley_call(self):
        self.match.rally((3.0, 12.0), (0.2, 3.0))
        self.match.idle(2)
        self.assertEqual(self.calls("singles")[0].call, "out")
        self.assertEqual(self.calls("doubles")[0].call, "in")

    def test_close_call_is_flagged_for_review(self):
        self.match.rally((3.0, 12.0), (BADMINTON_SINGLES_MARGIN + 0.05, 4.0))
        self.match.idle(2)
        call = self.calls()[0]
        self.assertEqual(call.call, "in")
        self.assertEqual(call.confidence, "medium")
        self.assertTrue(any("close call" in flag for flag in call.flags))

    def test_landing_beside_the_net_flags_uncertain_side(self):
        self.match.rally((3.0, 2.0), (3.0, 6.5))
        self.match.idle(2)
        call = self.calls()[0]
        self.assertEqual(call.call, "in")
        self.assertTrue(any("from the net" in flag for flag in call.flags))

    def test_landing_is_where_the_shuttle_touched_down_not_where_it_skidded_to(self):
        self.match.rally((3.0, 12.0), (3.0, 0.15), skid_to=(3.0, -0.25))  # in by 15 cm, rests 25 cm out
        self.match.idle(2)
        call = self.calls()[0]
        self.assertEqual((call.landing_method, call.call), ("settled", "in"))
        self.assertAlmostEqual(call.landing_court[1], 0.15, delta=0.08)

    def test_a_fast_slide_after_touching_down_is_not_taken_for_flight(self):
        # Lands 0.8 m inside the near baseline and slides 1.8 m in 0.2 s, fast enough to count as moving.
        self.match.rally((3.0, 2.0), (3.0, 12.6), skid_to=(3.0, 14.4), skid_frames=6)
        self.match.idle(2)
        call = self.calls()[0]
        self.assertEqual((call.landing_method, call.call), ("settled", "in"))
        self.assertAlmostEqual(call.landing_court[1], 12.6, delta=0.3)

    def test_shuttle_lost_on_its_way_down_lands_where_it_comes_to_rest(self):
        self.match.rally((3.0, 2.0), (3.0, 13.8), lost_from=0.8)   # out by 0.4 m, last seen high in the air
        self.match.idle(2)
        call = self.calls()[0]
        self.assertEqual((call.landing_method, call.call), ("settled", "out"))
        self.assertAlmostEqual(call.landing_court[1], 13.8, delta=0.05)

    def test_a_landing_far_outside_the_court_is_not_trusted(self):
        self.match.rally((3.0, 12.0), (3.0, -4.0))   # no shot lands 4 m long: the shuttle was in the air
        self.match.idle(2)
        # Keep the still shuttle, which would otherwise be dropped as a fixed object off the court.
        call = compute_line_calls(METADATA, self.match.records, LineCallConfig(static_share=1.0))[0]
        self.assertEqual((call.landing_method, call.call, call.confidence), ("settled", "out", "low"))
        self.assertIsNone(call.landing_error_m)
        self.assertTrue(any("still in the air" in flag for flag in call.flags))

    def test_landing_error_is_larger_far_from_the_camera_and_absent_when_lost(self):
        self.match.rally((3.0, 12.0), (3.0, 1.0))   # far court: fewer pixels per meter
        self.match.idle(3)
        self.match.rally((3.0, 2.0), (3.0, 12.4))   # near court
        self.match.idle(3)
        self.match.rally((3.0, 12.0), (3.0, 3.0), settle_seconds=0)
        self.match.idle(2)
        far, near, lost = self.calls()
        self.assertEqual(len(far.landing_error_m), 2)
        self.assertGreater(far.landing_error_m[1], near.landing_error_m[1])
        self.assertIsNone(lost.landing_error_m)

    def test_lost_shuttle_uses_last_position_with_low_confidence(self):
        self.match.rally((3.0, 2.0), (3.0, 10.0), settle_seconds=0)
        self.match.idle(2)
        call = self.calls()[0]
        self.assertEqual(call.landing_method, "last_seen")
        self.assertEqual(call.confidence, "low")

    def test_shuttle_held_by_player_is_not_a_landing(self):
        self.match.rally((3.0, 2.0), (3.0, 10.0), held=True)
        self.match.idle(2)
        call = self.calls()[0]
        self.assertEqual(call.landing_method, "held")
        self.assertEqual(call.confidence, "low")

    def test_high_shot_above_the_frame_does_not_end_the_rally(self):
        # Clear rises out of the top of the frame, is unseen for 1.2 s, then comes down and lands.
        for y in range(400, -1, -40):
            self.match._add(shuttle=[640.0, float(y)])
        self.match.idle(1.2)
        self.match.rally((3.0, 3.0), (3.0, 1.0), flight_seconds=0.8)
        self.match.idle(2)
        calls = self.calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].landing_method, "settled")

    def test_shuttle_carried_slowly_is_not_a_rally(self):
        start = np.array(self.match._image((3.0, 10.0)))
        for i in range(90):
            self.match._add(shuttle=[float(start[0] + i * 0.5), float(start[1])])
        self.match.idle(2)
        self.assertEqual(self.calls(), [])

    def test_shuttle_tapped_low_between_points_is_not_a_rally(self):
        # Knocked back to the server along the floor; both players stand still.
        start = np.array(self.match._image((3.0, 12.0)))
        for i in range(30):
            point = [float(start[0] + i * 8), float(start[1] - i * 2)]
            self.match._add(shuttle=point, courts={"upper": (3.0, 3.0), "lower": (3.0, 11.0)})
        self.match.idle(2)
        self.assertEqual(self.calls(), [])

    def test_short_rally_with_still_players_is_kept_when_the_shuttle_goes_high(self):
        # A serve straight into a winner: the players barely move, but the serve goes high.
        frames_before = len(self.match.records)
        self.match.rally((3.0, 12.0), (3.0, 2.0))
        for record in self.match.records[frames_before:]:
            record["players"]["upper"] = {"image": None, "court": [3.0, 3.0], "speed": None, "hands": {}}
            record["players"]["lower"] = {"image": None, "court": [3.0, 11.0], "speed": None, "hands": {}}
        self.match.idle(2)
        self.assertEqual(len(self.calls()), 1)

    def test_fixed_false_detection_off_court_is_ignored(self):
        # A stadium light above the court the detector keeps reporting as the shuttle.
        light = [1100.0, 60.0]
        for _ in range(3):
            self.match.rally((3.0, 12.0), (3.0, 2.0), settle_seconds=0.0)
            for _ in range(3 * FPS):
                self.match._add(shuttle=light)

        calls = self.calls()
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertEqual((call.call, call.landing_side), ("in", "upper"))
            self.assertAlmostEqual(call.landing_court[1], 2.0, delta=0.3)


class LoadRunTests(unittest.TestCase):
    def test_tracknet_positions_replace_the_detectors_when_saved_next_to_the_detections(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as file:
                json.dump(METADATA, file)
            with open(os.path.join(folder, "detections.jsonl"), "w", encoding="utf-8") as file:
                for frame, shuttle in ((1, [10.0, 20.0]), (2, None), (3, [30.0, 40.0])):
                    file.write(json.dumps({"frame": frame, "shuttlecock": {"image": shuttle}}) + "\n")

            def shuttles(**options):
                return [record["shuttlecock"]["image"] for record in load_run(folder, **options)[1]]

            self.assertEqual(shuttles(), [[10.0, 20.0], None, [30.0, 40.0]])
            with open(os.path.join(folder, TRACKNET_CSV), "w", encoding="utf-8") as file:
                file.write("frame,x,y\n2,15,25\n3,31,41\n")
            self.assertEqual(shuttles(), [None, [15.0, 25.0], [31.0, 41.0]])
            self.assertEqual(shuttles(use_tracknet=False), [[10.0, 20.0], None, [30.0, 40.0]])


if __name__ == "__main__":
    unittest.main()

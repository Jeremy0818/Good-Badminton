import unittest

import numpy as np

from badminton_analysis.court.mapper import CourtMapper
from badminton_analysis.detection.shuttlecock import ShuttlecockTracker


class _Array:
    def __init__(self, values):
        self.values = np.array(values, dtype=float).reshape(-1, *np.shape(values)[1:])
        self.shape = self.values.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Boxes:
    def __init__(self, detections):
        self.xywh = _Array([[x, y, 8, 8] for x, y, _ in detections] or np.zeros((0, 4)))
        self.conf = _Array([conf for _, _, conf in detections])


class _Result:
    def __init__(self, detections):
        self.boxes = _Boxes(detections)


class ScriptedModel:
    """Stands in for the YOLO model: returns the next frame's (x, y, confidence) detections."""

    def __init__(self, frames):
        self.frames = iter(frames)

    def __call__(self, frame, **kwargs):
        return [_Result(next(self.frames))]


CORNERS = [(400, 250), (880, 250), (1000, 650), (280, 650)]
FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
LIGHT = (1100, 60, 0.9)


def shuttle_at(i):
    return (300 + 10 * (i % 40), 400, 0.5)


class StaticSpotTests(unittest.TestCase):
    def run_tracker(self, frames, mapper=None, **kwargs):
        tracker = ShuttlecockTracker(ScriptedModel(frames), **kwargs)
        tracker.court_mapper = mapper
        accepted = []
        for _ in frames:
            point = tracker.detect_ball(FRAME)
            accepted.append(tuple(tracker.update_trajectory(point)))
        return tracker, accepted

    def test_fixed_object_is_ignored_once_it_has_stayed_long_enough(self):
        frames = [[LIGHT, shuttle_at(i)] for i in range(60)]
        tracker, accepted = self.run_tracker(frames, static_min_frames=20)
        self.assertEqual(accepted[0], (1100, 60))          # locked onto the light at first
        self.assertEqual(len(tracker.static_cells), 1)
        self.assertEqual(accepted[-1], shuttle_at(59)[:2])  # then follows the real shuttle
        self.assertNotIn((1100, 60), accepted[25:])

    def test_fixed_object_off_court_is_banned_sooner(self):
        frames = [[LIGHT, shuttle_at(i)] for i in range(200)]
        tracker, accepted = self.run_tracker(frames, mapper=CourtMapper(CORNERS))
        self.assertEqual(len(tracker.static_cells), 1)     # light, far behind the far baseline
        self.assertNotIn((1100, 60), accepted[155:])
        self.assertEqual(accepted[-1], shuttle_at(199)[:2])

    def test_shuttle_resting_on_court_is_kept_longer(self):
        frames = [[(640, 500, 0.6)] for _ in range(200)]    # ~7 s still, inside the court
        tracker, _ = self.run_tracker(frames, mapper=CourtMapper(CORNERS))
        self.assertEqual(tracker.static_cells, set())

    def test_shuttle_resting_briefly_is_not_banned(self):
        frames = [[(640, 500, 0.6)] for _ in range(90)]     # 3 s still on the floor
        tracker, accepted = self.run_tracker(frames)
        self.assertEqual(tracker.static_cells, set())
        self.assertEqual(accepted[-1], (640, 500))


if __name__ == "__main__":
    unittest.main()

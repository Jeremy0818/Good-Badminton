import unittest

import numpy as np

from badminton_analysis.detection.tracknet import HEIGHT, SEQ, WIDTH, TrackNetTracker
from badminton_analysis.live.tracker import LiveTracker

METADATA = {
    "video": {"fps": 10.0, "width": 1280, "height": 720},
    "court": {"corners": [[400, 250], [880, 250], [1000, 650], [280, 650]], "roi_corners": [[0, 0], [1280, 720]],
              "mid_height": 450},
}
COURT = np.full((720, 1280, 3), 90, dtype=np.uint8)
CUT = np.zeros((720, 1280, 3), dtype=np.uint8)   # a broadcast cutting away from the court


class SteadyTrackNet(TrackNetTracker):
    """Stands in for the network: sees the shuttle at the same spot in every frame."""

    def __init__(self):
        self.background = None

    def _heatmaps(self, window):
        heatmaps = np.zeros((SEQ, HEIGHT, WIDTH), dtype=np.float32)
        heatmaps[:, 100:104, 200:206] = 1.0
        return heatmaps


class CountingPose:
    def __init__(self):
        self.calls = 0

    def detect_players(self, roi, x1, y1):
        self.calls += 1
        return [(640.0, 600.0)], {600.0: (630, 560)}, {600.0: (650, 560)}


class CutAwayTracker(LiveTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template = "cuts"

    def _shows_court(self, image):
        return bool(image.any())


class LiveTrackerTests(unittest.TestCase):
    def test_records_come_in_order_a_few_frames_late_and_tracked_once_the_background_is_ready(self):
        live = LiveTracker(METADATA, SteadyTrackNet(), background_every=0.1, first_background=2)
        early = [record["frame"] for frame in range(1, 10) for record in live.add_frame(frame, COURT)]
        # Frame 1 comes before the background and goes straight out; TrackNet's first window, frames 2-9,
        # finishes the 4 frames that no later window will see.
        self.assertEqual(early, [1, 2, 3, 4, 5])
        self.assertEqual(live.done_through, 5)
        records = live.finish()
        self.assertEqual([record["frame"] for record in records], [6, 7, 8, 9])
        self.assertEqual(live.done_through, 9)
        self.assertTrue(all(record["shuttlecock"]["image"] == [507.0, 255.0] for record in records))

    def test_a_cut_away_from_the_court_finishes_the_frames_waiting_for_trackNet(self):
        live = CutAwayTracker(METADATA, SteadyTrackNet(), background_every=0.1, first_background=1)
        self.assertEqual(live.add_frame(1, COURT) + live.add_frame(2, COURT), [])
        records = live.add_frame(3, CUT)
        self.assertEqual([record["frame"] for record in records], [1, 2])
        self.assertEqual(live.court_frames, 2)

    def test_frames_between_poses_keep_the_players_of_the_last_pose(self):
        pose = CountingPose()
        live = LiveTracker(METADATA, SteadyTrackNet(), pose=pose, pose_every=0.3)
        records = [record for frame in range(1, 8) for record in live.add_frame(frame, COURT)] + live.finish()
        self.assertEqual(pose.calls, 3)   # frames 1, 4 and 7
        self.assertTrue(all(record["players"]["lower"]["hands"]["right"] == [650.0, 560.0] for record in records))


if __name__ == "__main__":
    unittest.main()

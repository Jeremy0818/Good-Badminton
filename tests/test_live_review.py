import json
import os
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

import cv2

from badminton_analysis.analysis.line_calls import RallyCall
from badminton_analysis.live.live_review import LiveReview, clip_times, cut_clip, touchdown_track

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
CORNERS = [[100, 80], [220, 80], [260, 200], [60, 200]]


def rally(number, start_frame, landing_frame, call="in", side="lower"):
    return RallyCall(rally=number, start_frame=start_frame, end_frame=landing_frame - 3,
                     start_sec=(start_frame - 1) / 30, end_sec=(landing_frame - 4) / 30, landing_frame=landing_frame,
                     landing_image=[150.0, 190.0], landing_court=[3.0, 12.0], landing_side=side, call=call,
                     margin_m=0.5, landing_error_m=[0.05, 0.08], landing_method="settled", confidence="high")


class ClipTests(unittest.TestCase):
    def test_clip_runs_from_shortly_before_the_rally_ends_to_after_the_landing(self):
        start, end = clip_times(rally(1, 31, 301), 30)
        self.assertAlmostEqual(start, 300 / 30 - 3.0)
        self.assertAlmostEqual(end, 300 / 30 + 1.5)
        self.assertEqual(clip_times(rally(2, 271, 301), 30)[0], 9.0)   # a short rally starts its clip

    def test_track_is_taken_around_the_touch_down(self):
        points = {frame: [float(frame), 1.0] for frame in range(1, 400)}
        track = touchdown_track(points, rally(1, 31, 301), 30)
        self.assertEqual(track[0], [(301 - 24 - 1) / 30, 277.0, 1.0])
        self.assertEqual(track[-1], [(301 + 18 - 1) / 30, 319.0, 1.0])

    @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
    def test_clip_is_cut_from_a_recording_still_being_written(self):
        folder = tempfile.mkdtemp()
        source = os.path.join(folder, "source.mp4")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", "testsrc=size=320x240:rate=30", "-t", "4", "-c:v", "mpeg4", "-g", "15",
                        "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", source], check=True)
        stream = os.path.join(folder, "stream.mp4")
        with open(source, "rb") as file:
            data = file.read()
        with open(stream, "wb") as file:
            file.write(data[:len(data) * 3 // 4])   # the last second has not arrived yet
        target = os.path.join(folder, "clip.mp4")
        self.assertIsNone(cut_clip(stream, target, 0.5, 2.0, 720))
        video = cv2.VideoCapture(target)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        size = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video.release()
        self.assertAlmostEqual(frames, 45, delta=2)
        self.assertEqual(size, (320, 240))


class LiveReviewTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.recording = SimpleNamespace(
            folder=self.folder, fps=30, id="20261109-120000", calls=[rally(1, 31, 301), rally(2, 400, 700, "out")],
            tracks={1: [[9.9, 150.0, 180.0], [10.0, 150.0, 190.0]]},
            clipper=SimpleNamespace(clips={1: {"file": "rally-1.mp4", "start": 7.0, "end": 11.5, "error": None}}))
        self.review = LiveReview(self.recording, names=("Ana", "Bo"))

    def test_a_rally_is_shown_once_its_clip_is_cut_with_times_in_the_clip(self):
        state = self.review.state({"corners": CORNERS}, (320, 240), lambda number: f"/clip/{number}")
        self.assertTrue(state["live"])
        self.assertEqual(len(state["rallies"]), 1)   # rally 2's clip is still being cut
        shown = state["rallies"][0]
        self.assertEqual((shown["video"], shown["clip_start"], shown["clip_end"]), ("/clip/1", 0.0, 4.5))
        self.assertAlmostEqual(shown["landing_time"], 300 / 30 - 7.0, places=3)
        self.assertEqual(shown["track"], [[2.9, 150.0, 180.0], [3.0, 150.0, 190.0]])
        self.assertEqual(shown["predicted_side"], "upper")   # in on the near side: the far player won it
        self.assertEqual(state["video"], {"width": 320, "height": 240, "fps": 30, "name": "20261109-120000"})
        self.assertEqual(len(state["court_to_image"]), 3)

    def test_decisions_are_saved_next_to_the_recording(self):
        self.review.state({"corners": CORNERS}, (320, 240), str)
        self.review.decide(0, "far")
        with open(os.path.join(self.folder, "review.json"), encoding="utf-8") as file:
            self.assertEqual(json.load(file)["decisions"], {"1": "upper"})
        state = self.review.state({"corners": CORNERS}, (320, 240), str)
        self.assertEqual(state["rallies"][0]["decision"], "upper")
        self.assertEqual(state["rallies"][0]["sides"]["upper"], "Bo")
        with self.assertRaises(IndexError):
            self.review.decide(1, "near")   # not shown yet


if __name__ == "__main__":
    unittest.main()

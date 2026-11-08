import unittest

import numpy as np
import torch

from badminton_analysis.detection.tracknet import HEIGHT, SEQ, WIDTH, TrackNet, TrackNetTracker, frame_runs, heatmap_peak


class FirstWindowTracker(TrackNetTracker):
    """Stands in for the network: the first window sees the shuttle in all 8 frames, later ones see nothing."""

    def __init__(self):
        self.background = np.zeros((3, HEIGHT, WIDTH), dtype=np.float32)
        self.windows = []

    def _heatmaps(self, window):
        self.windows.append(len(window))
        heatmaps = np.zeros((SEQ, HEIGHT, WIDTH), dtype=np.float32)
        if len(self.windows) == 1:
            heatmaps[:, 100:104, 200:206] = 1.0
        return heatmaps


class TrackNetTests(unittest.TestCase):
    def test_layer_names_match_the_trackNetV3_checkpoint(self):
        state = TrackNet().state_dict()
        self.assertEqual(len(state), 104)  # entries in TrackNetV3's TrackNet_best.pt
        for name in ("down_block_1.conv_1.conv.weight", "bottleneck.conv_3.bn.running_var", "predictor.bias"):
            self.assertIn(name, state)
        self.assertEqual(tuple(state["down_block_1.conv_1.conv.weight"].shape), (64, 27, 3, 3))

    def test_one_heatmap_per_frame(self):
        model = TrackNet().eval()
        with torch.no_grad():
            heatmaps = model(torch.rand(1, 27, 32, 64))
        self.assertEqual(tuple(heatmaps.shape), (1, 8, 32, 64))
        self.assertTrue(0 <= float(heatmaps.min()) and float(heatmaps.max()) <= 1)

    def test_peak_is_the_centre_of_the_largest_blob_in_source_pixels(self):
        heatmap = np.zeros((288, 512), dtype=np.float32)
        heatmap[100:104, 200:206] = 0.9   # shuttle
        heatmap[10, 10] = 0.9             # speck
        self.assertEqual(heatmap_peak(heatmap, 3.75, 3.75), [761, 382])
        self.assertIsNone(heatmap_peak(np.zeros((288, 512), dtype=np.float32), 3.75, 3.75))

    def test_each_frame_averages_the_windows_that_saw_it_trusting_their_middle(self):
        tracker = FirstWindowTracker()
        points = list(tracker.track_run([np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)] * 10))
        self.assertEqual(tracker.windows, [8, 8, 8])   # windows start at frames 0, 4 and 8; the last is padded
        # Frames 4-7 are also in the second window, which sees nothing. The first window wins where the frame
        # was nearer its middle: weight 4 against 1 and 3 against 2 keep frames 4 and 5, 2 and 1 lose 6 and 7.
        self.assertEqual(points, [[203, 102]] * 6 + [None] * 4)

    def test_frames_split_into_unbroken_runs(self):
        self.assertEqual(list(frame_runs([1, 2, 3, 7, 8, 10])), [(1, 3), (7, 8), (10, 10)])


if __name__ == "__main__":
    unittest.main()

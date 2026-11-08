"""Shuttle tracking with TrackNetV3 (https://github.com/qaz812345/TrackNetV3, MIT licence).

TrackNet looks at 8 frames at a time together with the video's median background and draws a
heatmap of the shuttle in each frame. It follows the small, blurred shuttle near the floor better
than the single-frame detector, which is what finding the touch-down depends on.

Weights: download TrackNetV3's checkpoints (link in its README) and put TrackNet_best.pt in weights/.

    python -m badminton_analysis.detection.tracknet outputs/<video_name>

writes tracknet.csv next to detections.jsonl. Line calls and the review use it when it is there.

The network below follows TrackNetV3's model code so its checkpoints load unchanged, under this licence:

    MIT License

    Copyright (c) 2024 qaz812345

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software, pretrained model checkpoints, and associated documentation
    files (the "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to whom
    the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
from torch import nn

from ..analysis.line_calls import TRACKNET_CSV, load_run

SEQ = 8                    # frames per prediction
HEIGHT, WIDTH = 288, 512   # network input size
DEFAULT_WEIGHTS = "weights/TrackNet_best.pt"
# A window starts every STRIDE frames, so each frame is seen by two windows, and its heatmaps are averaged
# with TrackNetV3's weights, which trust the middle of a window most. With no overlap, positions jump when
# the window edges move. On four ShuttleSet matches this put the median landing 27 cm from the label, against
# 30 cm with no overlap; a window at every frame, twice the work again, was no better.
STRIDE = 4
WINDOW_WEIGHTS = np.array([1, 2, 3, 4, 4, 3, 2, 1], dtype=np.float32)


class Conv2DBlock(nn.Module):
    """3x3 convolution, batch norm, ReLU."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvStack(nn.Module):
    """Conv2DBlocks in a row, named conv_1, conv_2, ... like TrackNetV3's Double2DConv / Triple2DConv."""

    def __init__(self, in_dim, out_dim, count):
        super().__init__()
        self.count = count
        for i in range(count):
            setattr(self, f"conv_{i + 1}", Conv2DBlock(in_dim if i == 0 else out_dim, out_dim))

    def forward(self, x):
        for i in range(self.count):
            x = getattr(self, f"conv_{i + 1}")(x)
        return x


class TrackNet(nn.Module):
    """TrackNetV3's U-Net. Input: background + 8 frames as 27 RGB channels; output: 8 heatmaps."""

    def __init__(self, in_dim=(SEQ + 1) * 3, out_dim=SEQ):
        super().__init__()
        self.down_block_1 = ConvStack(in_dim, 64, 2)
        self.down_block_2 = ConvStack(64, 128, 2)
        self.down_block_3 = ConvStack(128, 256, 3)
        self.bottleneck = ConvStack(256, 512, 3)
        self.up_block_1 = ConvStack(768, 256, 3)
        self.up_block_2 = ConvStack(384, 128, 2)
        self.up_block_3 = ConvStack(192, 64, 2)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))

    def forward(self, x):
        pool = lambda t: nn.functional.max_pool2d(t, 2)
        up = lambda t: nn.functional.interpolate(t, scale_factor=2)
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(pool(x1))
        x3 = self.down_block_3(pool(x2))
        x = self.bottleneck(pool(x3))
        x = self.up_block_1(torch.cat([up(x), x3], dim=1))
        x = self.up_block_2(torch.cat([up(x), x2], dim=1))
        x = self.up_block_3(torch.cat([up(x), x1], dim=1))
        return torch.sigmoid(self.predictor(x))


def heatmap_peak(heatmap, scale_x, scale_y, threshold=0.5):
    """Centre of the largest blob above the threshold, in source pixels; None if there is none.

    Same rule as TrackNetV3's predict.py: whole heatmap pixels, then scaled.
    """
    mask = (heatmap > threshold).astype(np.uint8)
    if not mask.any():
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max((cv2.boundingRect(c) for c in contours), key=lambda rect: rect[2] * rect[3])
    return [int(int(x + w / 2) * scale_x), int(int(y + h / 2) * scale_y)]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class TrackNetTracker:
    """Finds the shuttle in runs of consecutive frames. Call set_background before track_run."""

    def __init__(self, weights=DEFAULT_WEIGHTS, device=None):
        if not os.path.exists(weights):
            raise FileNotFoundError(f"TrackNet weights not found: {weights}. Download TrackNetV3's checkpoints "
                                    "(see https://github.com/qaz812345/TrackNetV3) and put TrackNet_best.pt in weights/.")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        params = checkpoint.get("param_dict", {})
        if params.get("seq_len", SEQ) != SEQ or params.get("bg_mode", "concat") != "concat":
            raise ValueError(f"{weights} is not an 8-frame TrackNet with background: {params}")
        self.device = device or pick_device()
        self.model = TrackNet().to(self.device).eval()
        self.model.load_state_dict(checkpoint["model"])
        # Half precision on the GPU: on an M3 Pro a window takes 57 ms instead of 74, with the same shuttle
        # positions on 112 broadcast frames and 2 minutes of live tracking.
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model.to(self.dtype)
        self.background = None

    @staticmethod
    def prepare(frame_bgr):
        """Network input for one frame: 512x288 RGB, channels first, 0..1.

        INTER_AREA averages like the PIL resize the model was trained with, at a fraction of the cost.
        """
        small = cv2.resize(frame_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)[..., ::-1]
        return np.ascontiguousarray(np.moveaxis(small, -1, 0))

    def set_background(self, frames_bgr):
        """Median of frames spread over the video: the court without the moving shuttle."""
        self.set_background_prepared([self.prepare(frame) for frame in frames_bgr])

    def set_background_prepared(self, frames):
        """set_background for frames already passed through prepare."""
        median = np.median(np.array(frames), axis=0).astype(np.uint8)
        self.background = median.astype(np.float32) / 255.0

    def _heatmaps(self, window):
        """The network's heatmaps for a window of 8 prepared frames."""
        stack = np.concatenate([self.background] + window, axis=0)
        with torch.no_grad():
            return self.model(torch.from_numpy(stack)[None].to(self.device, self.dtype)).float().cpu().numpy()[0]

    def track_run(self, frames_bgr):
        """Yield the shuttle [x, y] in source pixels, or None where it is not seen, for each frame of an
        unbroken run of frames. Takes any iterable, so a video can be read as it goes."""
        run = TrackNetRun(self)
        for image in frames_bgr:
            yield from run.push(image)
        yield from run.close()


class TrackNetRun:
    """One unbroken run of frames, fed to TrackNet as they arrive. A frame is finished once no later
    window will see it, 4 to 7 frames after it arrives."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.pending, self.heat, self.weight = [], [], []   # frames not finished yet, oldest first
        self.scale = None

    def push(self, image_bgr):
        """Add the next frame. Returns the shuttle [x, y] or None for each frame this finishes, oldest first."""
        self.scale = (image_bgr.shape[1] / WIDTH, image_bgr.shape[0] / HEIGHT)
        self.pending.append(self.tracker.prepare(image_bgr).astype(np.float32) / 255.0)
        self.heat.append(np.zeros((HEIGHT, WIDTH), np.float32))
        self.weight.append(0.0)
        return self._window() if len(self.pending) == SEQ else []

    def close(self):
        """Finish the frames left at the end of the run."""
        points = []
        while self.pending:
            points += self._window()
        return points

    def _window(self):
        window = self.pending + [self.pending[-1]] * (SEQ - len(self.pending))   # repeat the last frame at the end
        maps = self.tracker._heatmaps(window)
        for i in range(len(self.pending)):
            self.heat[i] += WINDOW_WEIGHTS[i] * maps[i]
            self.weight[i] += WINDOW_WEIGHTS[i]
        # The next window starts STRIDE frames on, so no other window will see the first STRIDE frames.
        points = []
        for _ in range(min(STRIDE, len(self.pending))):
            self.pending.pop(0)
            points.append(heatmap_peak(self.heat.pop(0) / self.weight.pop(0), *self.scale))
        return points


def frame_runs(frames):
    """(first, last) of each stretch of consecutive numbers in sorted frame numbers."""
    first = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            yield first, previous
            first = frame
        previous = frame
    yield first, previous


def track_video(video_path, frames, tracker, background_samples=300):
    """Yield (frame, [x, y] or None) for the given frame numbers, counted from 1 like detections.jsonl.

    Frames in between are skipped. The background is the median of the given frames only, so replays and
    other shots left out of the analysis do not spoil it.
    """
    frames = sorted(set(frames))
    capture = cv2.VideoCapture(video_path)
    samples = []
    for frame in np.array(frames)[np.linspace(0, len(frames) - 1, background_samples).astype(int)]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame) - 1)
        ok, image = capture.read()
        if ok:
            samples.append(image)
    tracker.set_background(samples)
    capture.release()

    # Decode from the start instead of seeking, which is unreliable on some recordings.
    capture = cv2.VideoCapture(video_path)
    position = 1   # number of the frame the next read returns

    def read(count):
        nonlocal position
        for _ in range(count):
            ok, image = capture.read()
            if not ok:
                return
            position += 1
            yield image

    for first, last in frame_runs(frames):
        while position < first and capture.grab():
            position += 1
        if position != first:
            break   # the video is shorter than the analysis says
        yield from zip(range(first, last + 1), tracker.track_run(read(last - first + 1)))
    capture.release()


def write_tracknet_csv(output_dir, weights=DEFAULT_WEIGHTS, video_path=None):
    """Track the shuttle in the analysed frames of a video and save tracknet.csv next to its detections."""
    metadata, records = load_run(output_dir, use_tracknet=False)
    video_path = video_path or metadata["video"]["path"]
    frames = sorted({record["frame"] for record in records})
    per_minute = int(float(metadata["video"]["fps"]) * 60)
    tracker = TrackNetTracker(weights)
    started = time.time()
    path = os.path.join(output_dir, TRACKNET_CSV)
    partial = path + ".part"   # renamed when complete, so a stopped run never leaves a half-filled tracknet.csv
    done = seen = 0
    with open(partial, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "x", "y"])
        for frame, point in track_video(video_path, frames, tracker):
            done += 1
            if point is not None:
                writer.writerow([frame, *point])
                seen += 1
            if done % per_minute == 0:
                print(f"{done // per_minute} of {len(frames) // per_minute} min in {time.time() - started:.0f}s", flush=True)
    os.replace(partial, path)
    print(f"Shuttle found in {seen} of {len(frames)} frames; saved {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Track the shuttle with TrackNetV3 for an analysed video")
    parser.add_argument("output_dir")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--video", default=None, help="Video file, if it has moved since the analysis")
    args = parser.parse_args()
    write_tracknet_csv(args.output_dir, args.weights, args.video)


if __name__ == "__main__":
    main()

"""Track a match while it is filmed.

Frames are handled as they arrive. Frames that do not show the court are skipped (broadcasts cut to
replays and close-ups), TrackNet finds the shuttle 4 to 7 frames after a frame arrives, and the
players' poses are taken a few times a second rather than on every frame. Each court frame gives one
record in the format of the main pipeline's detections.jsonl, so line calls and the review can read it.
Each rally's line call is made from the records as soon as it is final, about 2.5 s after the landing.

Try it on an analysed video, replayed at its own speed as if it were being filmed now:

    python -m badminton_analysis.live.tracker outputs/<video_name> --start 600 --seconds 120

It writes live_detections.jsonl and live_line_calls.json next to detections.jsonl, prints each call as
it is made, and reports how long each step takes per frame and how long after its frame each record
was ready.
"""

import argparse
import collections
import json
import os
import time
from dataclasses import asdict

import cv2
import numpy as np

from ..analysis.line_calls import LiveLineCaller
from ..court.mapper import CourtMapper
from ..data.writer import JsonlDetectionWriter
from ..detection.tracknet import DEFAULT_WEIGHTS, TrackNetRun, TrackNetTracker, pick_device
from ..tracking.player import PlayerTracker

LIVE_DETECTIONS = "live_detections.jsonl"
LIVE_CALLS = "live_line_calls.json"
COURT_MATCH = 0.75   # template match score of a frame that shows the court, as in the main pipeline
# The court check compares frames shrunk to a quarter of their size: on 3600 broadcast frames it picked the
# same court frames as the main pipeline's full-size check, in 1.6 ms a frame instead of 30.
COURT_SCALE = 4


class LiveTracker:
    """Turns frames into detection records as they arrive.

    TrackNet needs the court without the shuttle, which it gets from the median of court frames:
    one every background_every seconds until there are background_frames, then one every
    sample_every seconds. The first median is taken from first_background frames, so tracking starts
    within seconds, and taken again whenever the number of frames has doubled and every refresh_every
    seconds. Until the first median, records have no shuttle.

    Frames between poses keep the players of the last pose, so a hand holding the shuttle is seen on
    every frame, as in the main pipeline, which takes a pose on every frame.
    """

    def __init__(self, metadata, tracknet, pose=None, pose_every=0.2, background_every=0.2, first_background=10,
                 background_frames=50, sample_every=2.0, refresh_every=60.0, max_samples=150, court_every=0.0):
        video, court = metadata["video"], metadata["court"]
        self.fps = float(video["fps"])
        self.tracknet = tracknet
        self.pose = pose
        corners = [tuple(corner) for corner in court["corners"]]
        if pose is not None:
            pose.court_mapper = CourtMapper(corners)
        self.roi = court["roi_corners"]
        self.template = None
        if court.get("template_path"):   # a fixed phone always shows the court; a broadcast cuts away
            self.court_size = (int(video["width"]) // COURT_SCALE, int(video["height"]) // COURT_SCALE)
            template = cv2.imread(court["template_path"], cv2.IMREAD_GRAYSCALE)
            self.template = cv2.resize(template, self.court_size, interpolation=cv2.INTER_AREA)
        self.players = PlayerTracker(corners, threshold=court["mid_height"], detection_writer=self, fps=self.fps)
        self.pose_every = max(1, round(pose_every * self.fps))
        # A broadcast can cut away on any frame. A phone on a stand keeps its view, so checking a few times a
        # second is enough to notice it being moved.
        self.court_every = max(1, round(court_every * self.fps))
        self.court_checked, self.shows_court = None, True
        self.background_every = max(1, round(background_every * self.fps))
        self.first_background = first_background
        self.background_frames = background_frames
        self.sample_every = max(1, round(sample_every * self.fps))
        self.refresh_every = refresh_every * self.fps
        self.max_samples = max_samples
        self.samples, self.last_sample, self.last_refresh, self.refreshed_with = [], None, None, 0
        self.last_pose = ([], {}, {})
        self.run = None                        # TrackNet run of the current stretch of court frames
        self.waiting = collections.deque()     # records whose shuttle TrackNet has not given yet
        self.finished = []
        self.frame = 0
        self.court_frames = 0
        self.record = None
        self.times = collections.defaultdict(float)   # seconds spent in each step

    def write(self, record):
        """PlayerTracker hands over each record here."""
        self.record = record

    def add_frame(self, frame, image):
        """Process frame number `frame`, counted from 1. Returns the records finished so far, in order."""
        self.frame = frame
        started = time.perf_counter()
        if self.template is not None and (self.court_checked is None or frame - self.court_checked >= self.court_every):
            self.shows_court, self.court_checked = self._shows_court(image), frame
        self._time("court check", started)
        if not self.shows_court:
            self._end_run()
            return self._take()
        self.court_frames += 1
        if self.pose is not None and (self.court_frames - 1) % self.pose_every == 0:
            started = time.perf_counter()
            (x1, y1), (x2, y2) = self.roi
            self.last_pose = self.pose.detect_players(image[y1:y2, x1:x2], x1, y1)
            self._time("pose", started)
        centroids, left_hands, right_hands = self.last_pose
        self.players.update(frame, centroids, None, left_hands, right_hands, self.court_frames)
        started = time.perf_counter()
        self._keep_for_background(frame, image)
        self._time("background", started)
        if self.tracknet.background is None:
            self.finished.append(self.record)
        else:
            started = time.perf_counter()
            self.run = self.run or TrackNetRun(self.tracknet)
            self.waiting.append(self.record)
            self._fill(self.run.push(image))
            self._time("TrackNet", started)
        return self._take()

    def finish(self):
        """Records left at the end of the video."""
        self._end_run()
        return self._take()

    @property
    def done_through(self):
        """The frame up to which every record has been handed out: frames that do not show the court
        give none."""
        return self.waiting[0]["frame"] - 1 if self.waiting else self.frame

    def _shows_court(self, image):
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), self.court_size, interpolation=cv2.INTER_AREA)
        return float(np.max(cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED))) >= COURT_MATCH

    def _keep_for_background(self, frame, image):
        every = self.background_every if len(self.samples) < self.background_frames else self.sample_every
        if self.last_sample is not None and frame - self.last_sample < every:
            return
        self.last_sample = frame
        self.samples.append(self.tracknet.prepare(image))
        if len(self.samples) > self.max_samples:
            del self.samples[::2]   # keep them spread over the match
        if self.tracknet.background is None:
            due = len(self.samples) >= self.first_background
        else:
            due = len(self.samples) >= 2 * self.refreshed_with or frame - self.last_refresh >= self.refresh_every
        if due:
            self.tracknet.set_background_prepared(self.samples)
            self.last_refresh, self.refreshed_with = frame, len(self.samples)

    def _fill(self, points):
        for point in points:
            record = self.waiting.popleft()
            record["shuttlecock"]["image"] = None if point is None else [float(point[0]), float(point[1])]
            self.finished.append(record)

    def _end_run(self):
        if self.run is not None:
            started = time.perf_counter()
            self._fill(self.run.close())
            self.run = None
            self._time("TrackNet", started)

    def _take(self):
        records, self.finished = self.finished, []
        return records

    def _time(self, step, started):
        self.times[step] += time.perf_counter() - started


def replay(video_path, first_frame, last_frame, fps, speed=1.0, times=None):
    """Yield (frame, image, due) for frames first_frame to last_frame of a video, each at the time it
    would arrive if it were being filmed now (due, from time.perf_counter). Frames are never dropped,
    so a tracker that is too slow falls further and further behind. speed 0 replays as fast as possible."""
    capture = cv2.VideoCapture(video_path)
    for _ in range(first_frame - 1):   # decode up to the start: seeking is unreliable on some recordings
        if not capture.grab():
            return
    start = time.perf_counter()
    for frame in range(first_frame, last_frame + 1):
        started = time.perf_counter()
        ok, image = capture.read()
        if times is not None:
            times["decode"] += time.perf_counter() - started
        if not ok:
            break
        due = start + (frame - first_frame) / (fps * speed) if speed else time.perf_counter()
        wait = due - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        yield frame, image, due
    capture.release()


def main():
    parser = argparse.ArgumentParser(description="Track an analysed video as if it were being filmed now")
    parser.add_argument("output_dir")
    parser.add_argument("--start", type=float, default=0.0, help="Second of the video to start at")
    parser.add_argument("--seconds", type=float, default=120.0, help="Seconds of video to replay")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed; 0 for as fast as possible")
    parser.add_argument("--pose-every", type=float, default=0.2, help="Seconds between poses; 0 for none")
    parser.add_argument("--pose-model", default="weights/yolo11n-pose.pt", help="YOLO pose model, as in main.py")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--video", default=None, help="Video file, if it has moved since the analysis")
    parser.add_argument("--every-frame", action="store_true",
                        help="Track every frame, as for a phone that always shows the court: the load to keep up with")
    args = parser.parse_args()

    with open(os.path.join(args.output_dir, "metadata.json"), "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if args.every_frame:
        metadata["court"].pop("template_path", None)
    fps = float(metadata["video"]["fps"])
    pose = None
    if args.pose_every > 0:
        from ..detection.yolo_pose import YOLOPoseProcessor
        from ..visualization.player_pose import PlayerPoseVisualizer
        # On the GPU it takes half the time: 17 ms against 34 on an M3 Pro's CPU.
        processor = YOLOPoseProcessor(model_path=args.pose_model, device=pick_device())
        pose = PlayerPoseVisualizer(processor, show_skeletons=False, show_player_trajectories=False)
    live = LiveTracker(metadata, TrackNetTracker(args.weights), pose, pose_every=args.pose_every or 1.0)
    caller = LiveLineCaller(metadata)
    first = int(args.start * fps) + 1
    last = first + int(args.seconds * fps) - 1
    delays, calls = [], []
    first_due = None

    def report(new_calls):
        for call in new_calls:
            # How long after the landing frame was filmed the call was made; replaying as fast as possible
            # has no filming time.
            after = time.perf_counter() - first_due - (call.landing_frame - first) / (fps * args.speed) \
                if args.speed else None
            margin = abs(call.margin_m)
            distance = f"{margin * 100:.0f} cm" if margin < 1 else f"{margin:.2f} m"
            print(f"rally {call.rally}: {call.call.upper()} by {distance}, landed at frame {call.landing_frame}"
                  + (f", called {after:.1f} s later" if after is not None else ""), flush=True)
            calls.append(dict(asdict(call), called_after_s=after))

    with JsonlDetectionWriter(os.path.join(args.output_dir, LIVE_DETECTIONS)) as writer:
        due_at = {}
        started = None
        for frame, image, due in replay(args.video or metadata["video"]["path"], first, last, fps, args.speed,
                                        live.times):
            started = started or time.perf_counter()
            first_due = first_due or due
            due_at[frame] = due
            records = live.add_frame(frame, image)
            for record in records:
                writer.write(record)
                delays.append(time.perf_counter() - due_at.pop(record["frame"]))
            for old in [f for f in due_at if f < frame - 2 * fps]:   # frames that gave no record
                del due_at[old]
            report(caller.update(records, live.done_through / fps))
        records = live.finish()
        for record in records:
            writer.write(record)
            delays.append(time.perf_counter() - due_at.pop(record["frame"], time.perf_counter()))
        report(caller.update(records, live.done_through / fps) + caller.finish())
        elapsed = time.perf_counter() - (started or time.perf_counter())
    with open(os.path.join(args.output_dir, LIVE_CALLS), "w", encoding="utf-8") as file:
        json.dump(calls, file, indent=1)

    frames = last - first + 1
    steps = ", ".join(f"{step} {seconds / frames * 1000:.1f}" for step, seconds in live.times.items())
    print(f"{frames} frames ({frames / fps:.0f} s of video, {live.court_frames} showing the court) in {elapsed:.0f} s")
    print(f"ms per frame: {steps}; total {sum(live.times.values()) / frames * 1000:.1f} of {1000 / fps:.1f} available")
    if delays:
        print(f"records ready {np.median(delays):.2f} s after their frame (median), {max(delays):.2f} s at most")


if __name__ == "__main__":
    main()

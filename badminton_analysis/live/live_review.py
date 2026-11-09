"""Review each rally while the match is filmed.

The page of analysis/review_web.py, fed with the calls a phone has made so far. As each call is made, its rally's
clip is cut from the recording, which is still being written, so the rally can be watched with its touch-down
close-up a few seconds after the shuttle landed. Decisions and the score are saved next to the recording, as in
a review afterwards.
"""

import os
import queue
import subprocess
import threading

from ..analysis.review import load_session, save_session
from ..analysis.review_web import (CLIP_LEAD, CLIP_TAIL, TRACK_AFTER, TRACK_BEFORE, apply_choice, rally_state,
                                   review_state)
from ..court.mapper import CourtMapper

CLIPS = "clips"   # folder of the rallies' clips, next to the recording


def clip_times(call, fps):
    """Seconds of the recording a rally's clip covers: up to CLIP_LEAD before the rally ends, to CLIP_TAIL after.
    Frames count from 1, so a frame's time is (frame - 1) / fps."""
    end_frame = max(call.end_frame, call.landing_frame or 0)
    start = max((call.start_frame - 1) / fps, (end_frame - 1) / fps - CLIP_LEAD)
    return start, (end_frame - 1) / fps + CLIP_TAIL


def touchdown_track(points, call, fps):
    """The shuttle's positions around the touch-down as [seconds into the recording, x, y], from points by frame."""
    if not call.landing_frame:
        return []
    first, last = call.landing_frame - round(TRACK_BEFORE * fps), call.landing_frame + round(TRACK_AFTER * fps)
    return [[(frame - 1) / fps, *points[frame]] for frame in range(first, last + 1) if frame in points]


def cut_clip(source, target, start, end, height):
    """Re-encode seconds [start, end) of a recording, which may still be growing, as an MP4 that browsers can
    seek in, at most `height` pixels tall like the frames tracked. Returns ffmpeg's error, or None."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", source,
         "-t", f"{end - start:.3f}", "-map", "0:v:0", "-vf", f"scale=-2:min({height}\\,ih)", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", target],
        capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(target) and os.path.getsize(target) > 0:
        return None
    return (result.stderr or "no clip written").strip()[-300:]


class ClipCutter(threading.Thread):
    """Cuts each called rally's clip from a recording, in the order the rallies were called.

    source() gives the file to cut from: the growing stream, or the saved recording once it has replaced it."""

    def __init__(self, source, folder, fps, height):
        super().__init__(daemon=True)
        self.source, self.folder, self.fps, self.height = source, folder, fps, height
        self.jobs = queue.Queue()
        self.clips = {}   # rally -> {"file", "start", "end", "error"}

    def add(self, call):
        self.jobs.put(call)

    def run(self):
        os.makedirs(self.folder, exist_ok=True)
        while True:
            call = self.jobs.get()
            start, end = clip_times(call, self.fps)
            target = os.path.join(self.folder, f"rally-{call.rally}.mp4")
            error = cut_clip(self.source(), target, start, end, self.height)
            self.clips[call.rally] = {"file": target, "start": start, "end": end, "error": error}


class LiveReview:
    """A recording's rallies called so far, as review_web.html shows them. A rally is shown once its clip is cut;
    clips are cut in order, so the rallies shown are the first ones called."""

    def __init__(self, recording, names=("Player 1", "Player 2"), player0_side="lower", match_type="singles"):
        self.recording = recording   # capture's CaptureSession, with its calls, tracks and clips
        self.match_type = match_type
        self.review = load_session(recording.folder, [], player0_side, (0, 0), (0, 0), names)
        self.lock = threading.Lock()

    def _ready(self):
        clips = self.recording.clipper.clips if self.recording.clipper else {}
        ready = []
        for call in list(self.recording.calls):
            if call.rally not in clips:
                break
            ready.append(call)
        return ready

    def state(self, court, size, clip_url):
        """court: the corners the calls were made with; size: of the frames tracked, which the clips have too;
        clip_url(rally): where the page gets a rally's clip."""
        with self.lock:
            fps = self.recording.fps
            self.review.calls = self._ready()
            rows, match = self.review.replay()
            rallies = []
            for index, (call, row) in enumerate(zip(self.review.calls, rows)):
                clip = self.recording.clipper.clips[call.rally]
                start = clip["start"]
                landing = (call.landing_frame - 1) / fps if call.landing_frame else None
                rallies.append(rally_state(index, call, row, self.review, {
                    "video": None if clip["error"] else clip_url(call.rally),
                    "clip_start": 0.0,
                    "clip_end": round(clip["end"] - start, 3),
                    "landing_time": None if landing is None else round(landing - start, 3),
                    "track": [[round(t - start, 3), x, y] for t, x, y in self.recording.tracks.get(call.rally, [])],
                }))
            court_to_image = None
            if court:
                court_to_image = CourtMapper([tuple(corner) for corner in court["corners"]]).inv_matrix.tolist()
            video = {"width": size[0], "height": size[1], "fps": fps, "name": self.recording.id}
            return {**review_state(self.review, rallies, match, self.match_type, court_to_image, video), "live": True}

    def decide(self, index, choice):
        with self.lock:
            self.review.calls = self._ready()
            apply_choice(self.review, index, choice)
            save_session(self.recording.folder, self.review)

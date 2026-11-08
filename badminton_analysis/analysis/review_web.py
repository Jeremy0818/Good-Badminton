"""Review rally calls in the browser.

Same decisions and files as the OpenCV review (review.json, review_report.json),
served on localhost with the standard library only:

    python -m badminton_analysis.analysis.review_web outputs/<video_name> --names "Lee,Tan" --player1-side lower

Then open http://127.0.0.1:8765
"""

import argparse
import json
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..court.mapper import CourtMapper
from .line_calls import LineCallConfig, compute_line_calls, load_run
from .review import NOT_A_RALLY, load_session, save_session
from .scoring import parse_pair

PAGE = os.path.join(os.path.dirname(__file__), "review_web.html")
CLIP_LEAD = 3.0   # seconds of play shown before the rally ends
CLIP_TAIL = 1.5   # seconds shown after the landing
CHOICES = {"near": "lower", "far": "upper", "none": NOT_A_RALLY}
TRACK_BEFORE = 0.8  # seconds of shuttle positions sent for the touch-down close-up
TRACK_AFTER = 0.6


class ReviewApp:
    def __init__(self, output_dir, video_path, session, metadata, calls, match_type="singles", court_frames=(),
                 shuttle_points=None):
        self.output_dir = output_dir
        # Frames from the main court camera; broadcasts cut to close-ups soon after a rally.
        self.court_frames = set(court_frames)
        self.shuttle_points = shuttle_points or {}  # frame -> shuttle [x, y] in the image
        corners = metadata.get("court", {}).get("corners")
        self.court_to_image = CourtMapper([tuple(c) for c in corners]).inv_matrix.tolist() if corners else None
        self.match_type = match_type
        self.video_path = video_path
        self.session = session
        self.metadata = metadata
        self.calls = calls
        self.lock = threading.Lock()

    def state(self):
        video = self.metadata["video"]
        fps = float(video["fps"])
        rows, match = self.session.replay()
        rallies = []
        for index, (call, row) in enumerate(zip(self.calls, rows)):
            # Records count frames from 1; a frame's time in the video is (frame - 1) / fps.
            end_frame = max(call.end_frame, call.landing_frame or 0)
            start = max((call.start_frame - 1) / fps, (end_frame - 1) / fps - CLIP_LEAD)
            last = end_frame
            while last + 1 in self.court_frames and last < end_frame + CLIP_TAIL * fps:
                last += 1
            rallies.append({
                "index": index,
                "rally": call.rally,
                "clip_start": round(start, 3),
                "clip_end": round((last - 1) / fps, 3),
                "landing_time": round((call.landing_frame - 1) / fps, 3) if call.landing_frame else None,
                "landing_image": call.landing_image,
                "landing_court": call.landing_court,
                "landing_error_m": call.landing_error_m,
                "track": self._track(call.landing_frame, fps),
                "call": call.call,
                "margin_m": call.margin_m,
                "landing_side": call.landing_side,
                "landing_method": call.landing_method,
                "confidence": call.confidence,
                "flags": call.flags,
                "predicted_side": self.session.predicted_side(index),
                "decision": self.session.decisions.get(call.rally),
                **{key: row[key] for key in ("winner_side", "winner", "score_before", "score_after",
                                             "game", "sides", "reviewed", "changed")},
            })
        return {
            "names": list(self.session.names),
            "match_type": self.match_type,
            "court_to_image": self.court_to_image,  # homography for drawing court lines on the video
            "video": {"width": video["width"], "height": video["height"], "fps": fps,
                      "name": video.get("name", "")},
            "final_score": match.score_text(),
            "games": list(match.games),
            "points": list(match.points),
            "reviewed": sum(r["reviewed"] for r in rallies),
            "changed": sum(r["changed"] for r in rallies),
            "next_index": self.session.first_unreviewed(),
            "rallies": rallies,
        }

    def _track(self, frame, fps):
        """Shuttle positions around the touch-down as [time, x, y]."""
        if not frame:
            return []
        first, last = frame - round(TRACK_BEFORE * fps), frame + round(TRACK_AFTER * fps)
        return [[round((f - 1) / fps, 3), *self.shuttle_points[f]] for f in range(first, last + 1)
                if f in self.shuttle_points]

    def decide(self, index, choice):
        with self.lock:
            if choice == "accept":
                if not self.session.accept(index):
                    raise ValueError("no automatic call to accept")
            elif choice == "clear":
                self.session.decisions.pop(self.calls[index].rally, None)
            elif choice in CHOICES:
                self.session.decide(index, CHOICES[choice])
            else:
                raise ValueError(f"unknown choice: {choice}")
            save_session(self.output_dir, self.session)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status=200):
            self._send(status, json.dumps(data).encode("utf-8"), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(PAGE, "rb") as file:
                    self._send(200, file.read(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                with app.lock:
                    self._json(app.state())
            elif self.path == "/video":
                self._send_video()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/decide":
                return self._send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                app.decide(int(body["index"]), body["choice"])
            except (KeyError, ValueError, IndexError) as error:
                return self._json({"error": str(error)}, 400)
            with app.lock:
                self._json(app.state())

        def _send_video(self):
            # The browser seeks with Range requests, so serve byte ranges.
            size = os.path.getsize(app.video_path)
            start, end = 0, size - 1
            match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if match:
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else size - 1
                else:
                    start = size - int(match.group(2))
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(app.video_path)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(app.video_path, "rb") as file:
                file.seek(start)
                remaining = end - start + 1
                try:
                    while remaining > 0:
                        chunk = file.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the browser cancels ranges it no longer needs

    return Handler


def serve(output_dir, match_type="singles", player0_side="lower", start_points=(0, 0), start_games=(0, 0),
          names=("Player 1", "Player 2"), video_path=None, host="127.0.0.1", port=8765):
    metadata, records = load_run(output_dir)
    calls = compute_line_calls(metadata, records, LineCallConfig(match_type=match_type))
    session = load_session(output_dir, calls, player0_side, start_points, start_games, names)
    video_path = video_path or metadata["video"]["path"]
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Cannot find video: {video_path} (pass --video)")
    shuttle_points = {record["frame"]: [round(float(v), 1) for v in record["shuttlecock"]["image"][:2]]
                      for record in records if (record.get("shuttlecock") or {}).get("image")}
    app = ReviewApp(output_dir, video_path, session, metadata, calls, match_type,
                    court_frames=(record["frame"] for record in records), shuttle_points=shuttle_points)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"Reviewing {len(calls)} rallies from {output_dir}")
    print(f"Open http://{host}:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Review rally calls in the browser")
    parser.add_argument("output_dir")
    parser.add_argument("--video", default=None, help="Video file, if it has moved since the analysis")
    parser.add_argument("--match-type", choices=["singles", "doubles"], default="singles")
    parser.add_argument("--player1-side", choices=["upper", "lower"], default="lower",
                        help="Side player 1 is on when the video starts (upper = far, lower = near)")
    parser.add_argument("--start-score", default="0-0", help="Points at the start of the video, player 1 first")
    parser.add_argument("--start-games", default="0-0", help="Games won at the start of the video, player 1 first")
    parser.add_argument("--names", default="Player 1,Player 2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.output_dir, args.match_type, args.player1_side, parse_pair(args.start_score),
          parse_pair(args.start_games), tuple(args.names.split(",", 1)), args.video, args.host, args.port)


if __name__ == "__main__":
    main()

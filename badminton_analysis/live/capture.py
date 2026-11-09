"""Film with a phone, record and decode on this computer.

The phone opens a page served from here and films in the browser (nothing to
install). It uploads the recording in one-second pieces over Wi-Fi. This
computer saves the recording and decodes it while it arrives:

    python -m badminton_analysis.live.capture

Open http://127.0.0.1:8080 on this computer. It shows the address to open on
the phone. Recordings go to recordings/<date-time>/recording.mp4.

Several phones can film at once, each from its own angle: open the same
address on each. Each phone's court is marked and tracked on its own, and the
calls of phones filming together are combined (analysis/multi_camera.py).
"""

import argparse
import collections
import datetime
import json
import os
import secrets
import shutil
import socket
import ssl
import queue
import subprocess
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ..analysis.line_calls import LiveLineCaller
from ..analysis.multi_camera import COMBINED_CALLS, combine, write_combined
from ..court.mapper import resolve_court_corners
from ..data.writer import JsonlDetectionWriter
from .tracker import LIVE_CALLS, LIVE_DETECTIONS, LiveTracker

HERE = os.path.dirname(__file__)
PHONE_PAGE = os.path.join(HERE, "capture_phone.html")
WATCH_PAGE = os.path.join(HERE, "capture_watch.html")
MAX_CHUNK = 64 << 20      # Safari can send oversized pieces after the camera was paused
PREVIEW_EVERY = 0.5       # seconds between preview images
# Frames are decoded at most this tall for the preview, marking the court and live tracking; the recording
# keeps its full size. TrackNet sees 512x288 anyway, and at 1080p the tracker kept up only just.
DECODE_HEIGHT = 720
# A shift of the picture this many pixels at DECODE_HEIGHT since the court was marked moves a landing by about
# 12 cm near the far baseline (Marin v Yigit's broadcast view), so the page asks for the court to be marked again.
MOVED_PX = 2.0


def lan_address():
    """Address other devices on the network reach this computer at."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("8.8.8.8", 80))  # picks the outgoing interface; sends nothing
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def local_hostname():
    """Bonjour name, which keeps working when the network hands out a new address."""
    return socket.gethostname().split(".")[0] + ".local"


def ensure_certificate(folder, names):
    """Self-signed certificate for this computer. The phone asks once to trust it.

    Phones only accept the certificate when the address in the browser is listed
    in it, so it is remade when this computer shows up under a new address.
    """
    os.makedirs(folder, exist_ok=True)
    cert, key = os.path.join(folder, "cert.pem"), os.path.join(folder, "key.pem")
    names_path = os.path.join(folder, "names.json")
    try:
        with open(names_path, "r", encoding="utf-8") as file:
            known = json.load(file)
    except (OSError, ValueError):
        known = []
    if os.path.exists(cert) and os.path.exists(key) and set(names) <= set(known):
        return cert, key
    names = sorted(set(known) | set(names))
    alt = ",".join(f"IP:{name}" if name.replace(".", "").isdigit() else f"DNS:{name}" for name in names)
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "397",
         "-keyout", key, "-out", cert, "-subj", "/CN=Good-Badminton capture",
         "-addext", f"subjectAltName={alt}", "-addext", "extendedKeyUsage=serverAuth"],
        check=True, capture_output=True)
    with open(names_path, "w", encoding="utf-8") as file:
        json.dump(names, file)
    return cert, key


def access_key(folder):
    """Short secret in the phone's address, so others on the Wi-Fi cannot upload."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "access_key.txt")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as file:
            file.write(secrets.token_urlsafe(5))
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


class CaptureSession:
    """One recording from the phone, saved piece by piece."""

    def __init__(self, folder, info):
        self.id = os.path.basename(folder)
        self.folder = folder
        self.info = info  # camera, size, frame rate and format reported by the phone
        extension = ".webm" if "webm" in str(info.get("mime", "")) else ".mp4"
        self.path = os.path.join(folder, "stream" + extension)
        self.saved_path = None
        self.error = None
        self.next_seq = 0
        self.bytes = 0
        self.gaps = 0
        self.started = time.time()
        self.last_piece = None
        self.stopped = None
        self.live = None   # LiveCalls once the court has been marked
        self.calls = []    # line calls made so far
        self.lock = threading.Lock()
        open(self.path, "wb").close()
        self.write_info()

    @property
    def fps(self):
        try:
            return max(1, min(240, round(float(self.info.get("fps") or 30))))
        except (TypeError, ValueError):
            return 30

    def append(self, seq, data):
        """Save a piece. Returns False for a repeat of a piece already saved."""
        with self.lock:
            if seq < self.next_seq:
                return False  # the phone retried after a lost reply
            if seq > self.next_seq:
                self.gaps += 1  # pieces went missing; the decoder skips over the hole
            with open(self.path, "ab") as file:
                file.write(data)
            self.next_seq = seq + 1
            self.bytes += len(data)
            self.last_piece = time.time()
            return True

    def stop(self):
        with self.lock:
            if self.stopped is None:
                self.stopped = time.time()
        self.write_info()

    def stale(self, after=20.0):
        """No piece for `after` seconds: the phone's page was probably closed or lost the network."""
        return time.time() - (self.last_piece or self.started) > after

    def finish(self):
        """Rewrite the stream as a normal video file that players and the review page can seek in."""
        if self.bytes == 0:
            shutil.rmtree(self.folder, ignore_errors=True)  # started but nothing arrived
            return
        target = os.path.join(self.folder, "recording" + os.path.splitext(self.path)[1])
        result = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", self.path,
             "-c", "copy", "-movflags", "+faststart", target],
            capture_output=True, text=True)
        if result.returncode == 0 and os.path.getsize(target) > 0:
            self.saved_path = target
            try:
                os.remove(self.path)
            except OSError:
                pass  # Windows keeps it while the decoder still reads it
        else:
            self.error = (result.stderr or "could not save the recording").strip()[-500:]
            self.saved_path = self.path
        self.write_info()

    def summary(self):
        end = self.stopped or time.time()
        return {
            "id": self.id,
            "info": self.info,
            "bytes": self.bytes,
            "pieces": self.next_seq,
            "gaps": self.gaps,
            "seconds": round(end - self.started, 1),
            "since_last_piece": round(time.time() - self.last_piece, 1) if self.last_piece else None,
            "stopped": self.stopped is not None,
            "saved_path": self.saved_path,
            "error": self.error,
        }

    def write_info(self):
        with open(os.path.join(self.folder, "capture.json"), "w", encoding="utf-8") as file:
            json.dump({**self.summary(), "started": self.started, "fps": self.fps}, file, indent=2)


class StreamDecoder(threading.Thread):
    """Decode a recording while it is still being written.

    The growing file is fed to ffmpeg, which returns frames at a constant rate,
    so frame number / fps is the time in the recording. on_frame(index, yuv)
    gets each frame as an I420 array (use to_bgr for OpenCV).
    """

    def __init__(self, path, fps, on_frame, finished, ended=None):
        super().__init__(daemon=True)
        self.path = path
        self.fps = fps
        self.on_frame = on_frame
        self.finished = finished  # () -> True once no more data will be written
        self.ended = ended        # called after the last frame
        self.size = None
        self.frames = 0
        self.errors = collections.deque(maxlen=5)

    def run(self):
        process = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-probesize", "1000000", "-analyzeduration", "1000000",
             "-i", "pipe:0", "-map", "0:v:0", "-fps_mode", "cfr", "-r", str(self.fps),
             "-vf", f"scale=-2:min({DECODE_HEIGHT}\\,ih)", "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._feed, args=(process.stdin,), daemon=True).start()
        threading.Thread(target=self._collect_errors, args=(process.stderr,), daemon=True).start()
        try:
            self._read(process.stdout)
        finally:
            process.stdout.close()
            process.wait()
            if self.ended:
                self.ended()

    def _read(self, stream):
        # YUV4MPEG: one header line with the size, then "FRAME\n" and the planes per frame.
        header = stream.readline().split()
        if not header:
            return
        width = next(int(token[1:]) for token in header if token.startswith(b"W"))
        height = next(int(token[1:]) for token in header if token.startswith(b"H"))
        self.size = (width, height)
        frame_bytes = width * height * 3 // 2
        while stream.readline():
            data = stream.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            self.frames += 1
            self.on_frame(self.frames, np.frombuffer(data, np.uint8).reshape(height * 3 // 2, width))

    def _feed(self, pipe):
        try:
            with open(self.path, "rb") as file:
                while True:
                    data = file.read(1 << 16)
                    if data:
                        pipe.write(data)
                    elif self.finished():
                        pipe.write(file.read())  # anything written just before the end
                        break
                    else:
                        time.sleep(0.05)
        except OSError:
            pass  # ffmpeg exited
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    def _collect_errors(self, stream):
        with stream:
            for line in stream:
                self.errors.append(line.decode("utf-8", "replace").strip())


def to_bgr(yuv):
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


# PyTorch's MPS backend is not safe to use from two threads at once, so with several phones their networks take
# turns; the rest of each phone's tracking still runs alongside.
GPU = threading.Lock()


def _one_at_a_time(function):
    def run(*args, **kwargs):
        with GPU:
            return function(*args, **kwargs)
    return run


def load_models():
    """TrackNet and the pose model, as the live tracker uses them. Each phone gets its own, as TrackNet keeps the
    phone's empty court picture; they are loaded when its court is first marked."""
    from ..detection.tracknet import DEFAULT_WEIGHTS, TrackNetTracker, pick_device
    from ..detection.yolo_pose import YOLOPoseProcessor
    from ..visualization.player_pose import PlayerPoseVisualizer
    processor = YOLOPoseProcessor(model_path="weights/yolo11n-pose.pt", device=pick_device())
    processor.process_frame = _one_at_a_time(processor.process_frame)
    pose = PlayerPoseVisualizer(processor, show_skeletons=False, show_player_trajectories=False)
    tracknet = TrackNetTracker(DEFAULT_WEIGHTS)
    tracknet._heatmaps = _one_at_a_time(tracknet._heatmaps)
    return tracknet, pose


class LiveCalls(threading.Thread):
    """Follows the shuttle on a recording's frames as they are decoded and makes each rally's line call.
    Writes live_detections.jsonl and live_line_calls.json next to the recording. Tracking a phone's
    frames leaves little time to spare, so it falls behind while the court picture TrackNet needs is
    first built; frames beyond `backlog` seconds behind are dropped rather than kept waiting."""

    def __init__(self, session, court, models, backlog=5.0, every_frame=False, on_calls=None):
        super().__init__(daemon=True)
        self.on_calls = on_calls   # called from this thread after each new call
        tracknet, pose = models
        tracknet.background = None   # left from an earlier recording
        metadata = {"video": {"fps": session.fps, "width": court["size"][0], "height": court["size"][1]},
                    "court": court}
        if every_frame:   # track frames that do not look like the marked court too, as a phone's always do
            metadata["court"] = {key: value for key, value in court.items() if key != "template_path"}
        self.session = session
        self.tracker = LiveTracker(metadata, tracknet, pose, court_every=0.2)
        self.caller = LiveLineCaller(metadata)
        self.frames = queue.Queue(maxsize=round(backlog * session.fps))
        self.dropped = 0
        self.error = None
        # How far the picture has shifted since the court was marked, measured every 2 s against the marked
        # picture by phase correlation, which the players running about hardly disturb.
        self.reference = self._small(cv2.imread(court["template_path"], cv2.IMREAD_GRAYSCALE))
        self.hanning = cv2.createHanningWindow(self.reference.shape[::-1], cv2.CV_32F)
        self.shift_every = round(2 * session.fps)
        self.shift = 0.0

    @staticmethod
    def _small(gray):
        return cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)

    def _measure_shift(self, image):
        (dx, dy), _ = cv2.phaseCorrelate(self.reference, self._small(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
                                         self.hanning)
        self.shift = 2 * float(np.hypot(dx, dy))

    def add(self, index, yuv):
        try:
            self.frames.put_nowait((index, yuv))
        except queue.Full:
            self.dropped += 1

    def end(self):
        self.frames.put(None)

    def state(self):
        return {"court_frames": self.tracker.court_frames, "behind": round(self.frames.qsize() / self.session.fps, 1),
                "dropped": self.dropped, "error": self.error, "shift_px": round(self.shift, 1),
                "moved": self.shift >= MOVED_PX}

    def run(self):
        fps, session = self.session.fps, self.session
        try:
            with JsonlDetectionWriter(os.path.join(session.folder, LIVE_DETECTIONS)) as writer:
                while True:
                    item = self.frames.get()
                    if item is None:
                        records = self.tracker.finish()
                        calls = self.caller.update(records, self.tracker.done_through / fps) + self.caller.finish()
                    else:
                        image = to_bgr(item[1])
                        records = self.tracker.add_frame(item[0], image)
                        calls = self.caller.update(records, self.tracker.done_through / fps)
                        if item[0] % self.shift_every == 0 and self.tracker.shows_court:
                            self._measure_shift(image)
                    for record in records:
                        writer.write(record)
                    for call in calls:
                        call.rally = len(session.calls) + 1   # numbered across a court marked again
                        session.calls.append(call)
                    if calls:
                        with open(os.path.join(session.folder, LIVE_CALLS), "w", encoding="utf-8") as file:
                            json.dump([asdict(call) for call in session.calls], file, indent=1)
                        if self.on_calls:
                            self.on_calls()
                    if item is None:
                        break
        except Exception as error:   # shown on the page; recording goes on
            self.error = f"tracking stopped: {error!r}"


class Phone:
    """A phone filming the court, or this computer's camera: its latest recording, and what is kept while it
    stays put, the court marked on its picture."""

    def __init__(self, phone_id, number):
        self.id = phone_id
        self.number = number
        self.name = f"Phone {number}"
        self.session = None       # its current or latest recording
        self.decoder = None
        self.preview = None       # JPEG of the latest decoded frame
        self.preview_time = 0.0
        self.frame = None         # the latest decoded frame, at preview time
        self.court_frame = None   # the frame the court is being marked on
        self.court = None         # corners etc. as in metadata.json's "court"; kept while the phone stays put
        self.models = None        # its TrackNet and pose, loaded when its court is first marked

    @property
    def recording(self):
        return self.session is not None and self.session.stopped is None


class CaptureApp:
    """Phones filming at the same time each record and are tracked on their own. Once two or more of them make
    calls, the calls are combined (analysis/multi_camera.py)."""

    def __init__(self, out_dir, key, phone_urls=(), every_frame=False):
        self.out_dir = out_dir
        self.every_frame = every_frame   # see LiveCalls
        self.key = key
        self.phone_urls = list(phone_urls)
        self.phones = {}         # by the id each phone's page keeps, in the order they first connected
        self.combined = None     # (calls, alignment, fps) of the phones filming together
        self.combine_error = None
        self.lock = threading.Lock()
        self.combine_lock = threading.Lock()

    def phone(self, phone_id=None):
        """The phone with this id; without one, the phone that started recording last."""
        if phone_id is None:
            if not self.phones:
                raise LookupError("no picture from a phone yet")
            started = [phone for phone in self.phones.values() if phone.session is not None]
            return max(started, key=lambda phone: phone.session.started) if started else next(iter(self.phones.values()))
        try:
            return self.phones[phone_id]
        except KeyError:
            raise LookupError(f"no phone {phone_id}") from None

    def start(self, info):
        with self.lock:
            phone_id = str(info.get("phone") or "phone")   # pages from before several phones send no id
            phone = self.phones.get(phone_id)
            if phone is None:
                phone = self.phones[phone_id] = Phone(phone_id, len(self.phones) + 1)
            for other in self.phones.values():
                if other.recording and (other is phone or other.session.stale()):
                    self._stop(other.session)   # started again without stopping, or its page was closed
            name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            folder = os.path.join(self.out_dir, name)
            suffix = 1
            while os.path.exists(folder):
                suffix += 1
                folder = os.path.join(self.out_dir, f"{name}-{suffix}")
            os.makedirs(folder)
            session = CaptureSession(folder, {**info, "phone_name": phone.name})
            phone.session = session
            phone.preview = None
            phone.decoder = StreamDecoder(session.path, session.fps,
                                          lambda index, yuv: self._on_frame(phone, session, index, yuv),
                                          finished=lambda: session.stopped is not None,
                                          ended=lambda: session.live and session.live.end())
            phone.decoder.start()
            if phone.court:
                self._track(phone, session)   # the phone is back where its court was marked; if not, mark it again
            return session.id

    def _track(self, phone, session):
        """Follow the shuttle and call the lines on this recording's frames from now on."""
        if session.live is not None:
            session.live.end()
        if phone.models is None:
            phone.models = load_models()
        session.live = LiveCalls(session, phone.court, phone.models, every_frame=self.every_frame,
                                 on_calls=self._combine)
        session.live.start()

    def _together(self):
        """The phones tracking now whose recordings overlap the first such phone's, first phone first."""
        tracked = [phone for phone in self.phones.values() if phone.session is not None and phone.session.live]
        if not tracked:
            return []
        first, now = tracked[0].session, time.time()
        return [phone for phone in tracked
                if phone.session.started < (first.stopped or now) and (phone.session.stopped or now) > first.started]

    def _combine(self):
        """Combine the calls of the phones filming together and save them next to the first one's recording.
        Called from each phone's tracking thread when it makes a call."""
        with self.combine_lock:
            try:
                together = self._together()
                if len(together) < 2:
                    return
                cameras = [{"name": phone.name, "calls": list(phone.session.calls), "fps": phone.session.fps,
                            "started": phone.session.started, "folder": phone.session.folder} for phone in together]
                config = together[0].session.live.caller.config
                calls, alignment = combine(cameras, config)
                write_combined(os.path.join(together[0].session.folder, COMBINED_CALLS), calls, alignment, cameras,
                               config)
                self.combined, self.combine_error = (calls, alignment, together[0].session.fps), None
            except Exception as error:   # shown on the page; each phone's own calls go on
                self.combine_error = f"combining stopped: {error!r}"

    def _session(self, session_id):
        for phone in self.phones.values():
            if phone.session is not None and phone.session.id == session_id:
                return phone.session
        raise LookupError(f"no recording {session_id}")

    def add_piece(self, session_id, seq, data):
        session = self._session(session_id)
        if session.stopped is not None:
            raise LookupError(f"recording {session_id} has ended")
        return session.append(seq, data)

    def stop(self, session_id):
        with self.lock:
            self._stop(self._session(session_id))

    def _stop(self, session):
        if session.stopped is None:
            session.stop()
            threading.Thread(target=session.finish, daemon=True).start()

    def _on_frame(self, phone, session, index, yuv):
        if session.live is not None:
            session.live.add(index, yuv)
        now = time.time()
        if session is not phone.session or now - phone.preview_time < PREVIEW_EVERY:
            return
        phone.preview_time = now
        frame = to_bgr(yuv)
        phone.frame = frame
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (640, max(2, round(height * 640 / width))), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            phone.preview = jpeg.tobytes()

    def preview(self, phone_id=None):
        """JPEG of the phone's latest picture, if there is one."""
        try:
            return self.phone(phone_id).preview
        except LookupError:
            return None

    def suggest_court(self, phone_id=None):
        """Hold the phone's latest picture to mark the court on, with the court's corners if they can be found."""
        phone = self.phone(phone_id)
        frame = phone.frame
        if frame is None:
            raise LookupError("no picture from the phone yet")
        phone.court_frame = frame
        corners, _, _ = resolve_court_corners(frame)
        height, width = frame.shape[:2]
        return {"size": [width, height], "corners": [[int(x), int(y)] for x, y in corners] if corners else None}

    def court_picture(self, phone_id=None):
        try:
            frame = self.phone(phone_id).court_frame
        except LookupError:
            return None
        if frame is None:
            return None
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes() if ok else None

    def set_court(self, corners, phone_id=None):
        """The corners of the doubles court on the phone's held picture, going round from one end of a baseline.
        The picture is kept too: the live tracker compares frames with it to notice when the view changes."""
        phone = self.phone(phone_id)
        frame = phone.court_frame
        if frame is None:
            raise LookupError("no picture held to mark the court on")
        if len(corners) != 4 or any(len(corner) != 2 for corner in corners):
            raise ValueError("expected 4 corners")
        corners, roi_corners, mid_height = resolve_court_corners(frame, manual_corners=corners)
        folder = phone.session.folder if phone.session else self.out_dir
        template = os.path.join(folder, "court.jpg")
        cv2.imwrite(template, frame)
        height, width = frame.shape[:2]
        phone.court = {"corners": [list(corner) for corner in corners], "roi_corners": [list(c) for c in roi_corners],
                       "mid_height": int(mid_height), "template_path": template, "size": [width, height]}
        with open(os.path.join(folder, "court.json"), "w", encoding="utf-8") as file:
            json.dump(phone.court, file, indent=2)
        if phone.recording:
            self._track(phone, phone.session)
        return phone.court

    def _phone_status(self, phone):
        session, decoder = phone.session, phone.decoder
        current = None
        if session is not None:
            current = session.summary()
            decoded = decoder.frames / session.fps if decoder else 0.0
            current.update({
                "fps": session.fps,
                "frames": decoder.frames if decoder else 0,
                "decoded_seconds": round(decoded, 1),
                # how far decoding trails the phone (upload + decode delay)
                "delay": round(max(0.0, (session.stopped or time.time()) - session.started - decoded), 1),
                "size": decoder.size if decoder else None,
                "decoder_errors": list(decoder.errors) if decoder else [],
                "tracking": session.live.state() if session.live else None,
                "calls": [{"rally": call.rally, "call": call.call, "margin_m": call.margin_m,
                           "landing_sec": round(call.landing_frame / session.fps, 1), "confidence": call.confidence}
                          for call in session.calls],
            })
        return {"id": phone.id, "name": phone.name, "recording": phone.recording, "session": current,
                "court": phone.court}

    def _combined_status(self):
        if self.combined is None:
            return {"calls": [], "phones": []}
        calls, alignment, fps = self.combined

        def error(value):
            return max(value) if value else None

        return {"phones": alignment, "calls": [{
            "rally": call.rally, "call": call.call, "margin_m": call.margin_m, "error_m": error(call.landing_error_m),
            "landing_sec": round(call.landing_frame / fps if call.landing_frame is not None else call.end_sec, 1),
            "confidence": call.confidence,
            # the combining's own notes; each phone's are on its own calls
            "flags": [flag for flag in call.flags if not any(flag.startswith(seen["camera"] + ": ")
                                                             for seen in call.cameras)],
            "phones": [{"name": seen["camera"], "call": seen["call"], "margin_m": seen["margin_m"],
                        "error_m": error(seen["landing_error_m"])} for seen in call.cameras],
        } for call in calls]}

    def status(self, session_id=None):
        """Every phone's recording and court, and the calls combined from the phones filming together.
        "session" and "court" are those of recording session_id (a phone's page asks for its own), else of the
        phone that started recording last."""
        phones = [self._phone_status(phone) for phone in self.phones.values()]
        current = next((item for item in phones if item["session"] and item["session"]["id"] == session_id), None)
        if current is None and phones:
            try:
                latest = self.phone().id
                current = next(item for item in phones if item["id"] == latest)
            except LookupError:
                pass
        return {"phone_urls": self.phone_urls, "recording": any(item["recording"] for item in phones),
                "phones": phones, "session": current and current["session"], "court": current and current["court"],
                "combined": self._combined_status(), "combine_error": self.combine_error,
                "recordings": self.recordings()}

    def recordings(self):
        items = []
        if not os.path.isdir(self.out_dir):
            return items
        for name in sorted(os.listdir(self.out_dir), reverse=True)[:20]:
            try:
                with open(os.path.join(self.out_dir, name, "capture.json"), "r", encoding="utf-8") as file:
                    info = json.load(file)
            except (OSError, ValueError):
                continue
            item = {key: info.get(key) for key in ("id", "seconds", "bytes", "saved_path", "error")}
            item["phone"] = (info.get("info") or {}).get("phone_name")
            items.append(item)
        return items


def make_handler(app, phone):
    """phone=True: the camera page and uploads for the phone, which must send the access key.
    phone=False: the status page on this computer (localhost only, no key)."""

    class Handler(BaseHTTPRequestHandler):
        timeout = 60

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

        def _allowed(self):
            if not phone:
                return True
            if secrets.compare_digest(self.headers.get("X-Key", ""), app.key):
                return True
            self._json({"error": "wrong access key; open the address shown on the computer"}, 403)
            return False

        def _page(self, path):
            with open(path, "rb") as file:
                self._send(200, file.read(), "text/html; charset=utf-8")

        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            if path in ("/", "/index.html"):
                self._page(PHONE_PAGE if phone else WATCH_PAGE)
            elif path == "/camera" and not phone:
                self._page(PHONE_PAGE)  # film with this computer's own camera, or ?test=1 for a test pattern
            elif path == "/api/status":
                if self._allowed():
                    self._json(app.status(query.get("s")))
            elif path == "/preview.jpg" and not phone:
                picture = app.preview(query.get("phone"))
                if picture:
                    self._send(200, picture, "image/jpeg")
                else:
                    self._send(404, b"no picture yet", "text/plain")
            elif path == "/court.jpg" and not phone:
                picture = app.court_picture(query.get("phone"))
                if picture:
                    self._send(200, picture, "image/jpeg")
                else:
                    self._send(404, b"no picture held", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._allowed():
                return
            url = urlparse(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_CHUNK:
                return self._json({"error": "piece too large"}, 413)
            body = self.rfile.read(length)
            try:
                if url.path == "/api/start":
                    info = json.loads(body or b"{}")
                    if not isinstance(info, dict):
                        raise ValueError("expected a JSON object")
                    self._json({"session": app.start(info)})
                elif url.path == "/api/piece":
                    saved = app.add_piece(query["s"], int(query["n"]), body)
                    self._json({"saved": saved})
                elif url.path == "/api/stop":
                    app.stop(query["s"])
                    self._json({"stopped": True})
                elif url.path == "/api/court/suggest" and not phone:
                    self._json(app.suggest_court(query.get("phone")))
                elif url.path == "/api/court" and not phone:
                    self._json({"court": app.set_court(json.loads(body)["corners"], query.get("phone"))})
                else:
                    self._send(404, b"not found", "text/plain")
            except LookupError as error:
                self._json({"error": str(error)}, 404)
            except (KeyError, ValueError) as error:
                self._json({"error": f"bad request: {error}"}, 400)

    return Handler


class TLSServer(ThreadingHTTPServer):
    """HTTPS server that does each TLS handshake in the connection's own thread."""

    def __init__(self, address, handler, context):
        super().__init__(address, handler)
        self.context = context

    def get_request(self):
        sock, address = self.socket.accept()
        sock.settimeout(60)
        return self.context.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), address

    def finish_request(self, request, client_address):
        try:
            request.do_handshake()
        except (ssl.SSLError, OSError):
            return  # usually the phone has not accepted the certificate yet
        super().finish_request(request, client_address)


def replay_as_phone(app, video, start=0.0, seconds=None, phone="replay"):
    """Send a saved video to the app as a phone would, at its own speed, to try the live steps without a
    phone. ffmpeg re-encodes it as the fragmented MP4 that Safari records, one fragment a second. Each phone
    id replays as a phone of its own."""
    probe = cv2.VideoCapture(video)
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    size = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-ss", str(start), "-i", video]
    if seconds:
        command += ["-t", str(seconds)]
    command += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-g", str(round(fps)), "-f", "mp4",
                "-movflags", "frag_keyframe+empty_moov+default_base_moof", "pipe:1"]
    session = app.start({"camera": f"replay of {os.path.basename(video)}", "width": size[0], "height": size[1],
                         "fps": fps, "mime": "video/mp4", "phone": phone})
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    seq = 0
    while data := process.stdout.read1(1 << 20):
        app.add_piece(session, seq, data)
        seq += 1
    process.wait()
    app.stop(session)


def serve(out_dir="recordings", phone_port=8443, local_port=8080, replays=(), every_frame=False):
    tls_dir = os.path.join(out_dir, ".tls")
    address, hostname = lan_address(), local_hostname()
    cert, key_file = ensure_certificate(tls_dir, [address, hostname, "localhost"])
    key = access_key(tls_dir)
    urls = [f"https://{host}:{phone_port}/?k={key}" for host in (hostname, address)]
    app = CaptureApp(out_dir, key, urls, every_frame=every_frame)

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key_file)
    phone_server = TLSServer(("0.0.0.0", phone_port), make_handler(app, phone=True), context)
    local_server = ThreadingHTTPServer(("127.0.0.1", local_port), make_handler(app, phone=False))
    threading.Thread(target=local_server.serve_forever, daemon=True).start()

    print(f"On this computer open http://127.0.0.1:{local_port}")
    print("On the phone (same Wi-Fi or this computer joined to the phone's hotspot) open:")
    for url in urls:
        print(f"  {url}")
    print("The phone warns that the connection is not private once: Show Details > visit this website.")
    print("Ctrl+C to stop")
    for number, replay in enumerate(replays, 1):
        threading.Thread(target=replay_as_phone, args=(app, *replay, f"replay {number}"), daemon=True).start()
    try:
        phone_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for phone in app.phones.values():
            if phone.recording:
                phone.session.stop()
                phone.session.finish()
        phone_server.server_close()
        local_server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Film with a phone; record and decode on this computer")
    parser.add_argument("--out", default="recordings", help="Folder for recordings")
    parser.add_argument("--phone-port", type=int, default=8443)
    parser.add_argument("--port", type=int, default=8080, help="Port of the status page on this computer")
    parser.add_argument("--replay", nargs="+", metavar="VIDEO",
                        help="Send these videos as if phones were filming them, a phone each, to try without a phone")
    parser.add_argument("--start", type=float, nargs="+", default=[0.0],
                        help="Second of the --replay videos to start at: one for all, or one each")
    parser.add_argument("--seconds", type=float, default=None, help="Seconds of the --replay videos to send")
    parser.add_argument("--every-frame", action="store_true",
                        help="Track every frame, even one that does not look like the marked court, as for a phone: "
                             "a --replay of a broadcast then puts on the load of a phone")
    args = parser.parse_args()
    replays = []
    if args.replay:
        starts = args.start * len(args.replay) if len(args.start) == 1 else args.start
        if len(starts) != len(args.replay):
            parser.error("give one --start for all --replay videos, or one each")
        replays = [(video, start, args.seconds) for video, start in zip(args.replay, starts)]
    serve(args.out, args.phone_port, args.port, replays, args.every_frame)


if __name__ == "__main__":
    main()

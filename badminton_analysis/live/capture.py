"""Film with a phone, record and decode on this computer.

The phone opens a page served from here and films in the browser (nothing to
install). It uploads the recording in one-second pieces over Wi-Fi. This
computer saves the recording and decodes it while it arrives:

    python -m badminton_analysis.live.capture

Open http://127.0.0.1:8080 on this computer. It shows the address to open on
the phone. Recordings go to recordings/<date-time>/recording.mp4.
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
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

HERE = os.path.dirname(__file__)
PHONE_PAGE = os.path.join(HERE, "capture_phone.html")
WATCH_PAGE = os.path.join(HERE, "capture_watch.html")
MAX_CHUNK = 64 << 20      # Safari can send oversized pieces after the camera was paused
PREVIEW_EVERY = 0.5       # seconds between preview images


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

    def __init__(self, path, fps, on_frame, finished):
        super().__init__(daemon=True)
        self.path = path
        self.fps = fps
        self.on_frame = on_frame
        self.finished = finished  # () -> True once no more data will be written
        self.size = None
        self.frames = 0
        self.errors = collections.deque(maxlen=5)

    def run(self):
        process = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-probesize", "1000000", "-analyzeduration", "1000000",
             "-i", "pipe:0", "-map", "0:v:0", "-fps_mode", "cfr", "-r", str(self.fps),
             "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._feed, args=(process.stdin,), daemon=True).start()
        threading.Thread(target=self._collect_errors, args=(process.stderr,), daemon=True).start()
        try:
            self._read(process.stdout)
        finally:
            process.stdout.close()
            process.wait()

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


class CaptureApp:
    def __init__(self, out_dir, key, phone_urls=()):
        self.out_dir = out_dir
        self.key = key
        self.phone_urls = list(phone_urls)
        self.session = None
        self.decoder = None
        self.preview = None   # JPEG of the latest decoded frame
        self.preview_time = 0.0
        self.lock = threading.Lock()

    def start(self, info):
        with self.lock:
            if self.session is not None and self.session.stopped is None:
                self._stop(self.session)  # the phone started again without stopping
            name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            folder = os.path.join(self.out_dir, name)
            suffix = 1
            while os.path.exists(folder):
                suffix += 1
                folder = os.path.join(self.out_dir, f"{name}-{suffix}")
            os.makedirs(folder)
            session = CaptureSession(folder, info)
            self.session = session
            self.preview = None
            self.decoder = StreamDecoder(session.path, session.fps, self._on_frame,
                                         finished=lambda: session.stopped is not None)
            self.decoder.start()
            return session.id

    def _session(self, session_id):
        session = self.session
        if session is None or session.id != session_id:
            raise LookupError(f"no recording {session_id}")
        return session

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

    def _on_frame(self, index, yuv):
        now = time.time()
        if now - self.preview_time < PREVIEW_EVERY:
            return
        self.preview_time = now
        frame = to_bgr(yuv)
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (640, max(2, round(height * 640 / width))), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.preview = jpeg.tobytes()

    def status(self):
        session, decoder = self.session, self.decoder
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
            })
        return {"phone_urls": self.phone_urls, "recording": bool(session and session.stopped is None),
                "session": current, "recordings": self.recordings()}

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
            items.append({key: info.get(key) for key in ("id", "seconds", "bytes", "saved_path", "error")})
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
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._page(PHONE_PAGE if phone else WATCH_PAGE)
            elif path == "/camera" and not phone:
                self._page(PHONE_PAGE)  # film with this computer's own camera, or ?test=1 for a test pattern
            elif path == "/api/status":
                if self._allowed():
                    self._json(app.status())
            elif path == "/preview.jpg" and not phone:
                if app.preview:
                    self._send(200, app.preview, "image/jpeg")
                else:
                    self._send(404, b"no picture yet", "text/plain")
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


def serve(out_dir="recordings", phone_port=8443, local_port=8080):
    tls_dir = os.path.join(out_dir, ".tls")
    address, hostname = lan_address(), local_hostname()
    cert, key_file = ensure_certificate(tls_dir, [address, hostname, "localhost"])
    key = access_key(tls_dir)
    urls = [f"https://{host}:{phone_port}/?k={key}" for host in (hostname, address)]
    app = CaptureApp(out_dir, key, urls)

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
    try:
        phone_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if app.session is not None and app.session.stopped is None:
            app.session.stop()
            app.session.finish()
        phone_server.server_close()
        local_server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Film with a phone; record and decode on this computer")
    parser.add_argument("--out", default="recordings", help="Folder for recordings")
    parser.add_argument("--phone-port", type=int, default=8443)
    parser.add_argument("--port", type=int, default=8080, help="Port of the status page on this computer")
    args = parser.parse_args()
    serve(args.out, args.phone_port, args.port)


if __name__ == "__main__":
    main()

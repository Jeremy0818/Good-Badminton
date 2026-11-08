import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from badminton_analysis.live.capture import (CaptureApp, CaptureSession, StreamDecoder, ensure_certificate,
                                             make_handler)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def fragmented_mp4(path, seconds=2, fps=30):
    """A short test video written the way phone browsers record: fragmented MP4."""
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=size=320x240:rate={fps}", "-t", str(seconds), "-c:v", "mpeg4",
                    "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", path], check=True)
    with open(path, "rb") as file:
        return file.read()


def pieces(data, count):
    step = len(data) // count + 1
    return [data[i:i + step] for i in range(0, len(data), step)]


class CaptureSessionTests(unittest.TestCase):
    def test_pieces_are_saved_in_order_and_repeats_are_ignored(self):
        session = CaptureSession(tempfile.mkdtemp(), {"mime": "video/mp4", "fps": 59.94})
        self.assertTrue(session.append(0, b"ab"))
        self.assertFalse(session.append(0, b"ab"))   # the phone retried after a lost reply
        self.assertTrue(session.append(1, b"cd"))
        self.assertTrue(session.append(3, b"ef"))    # piece 2 never arrived
        with open(session.path, "rb") as file:
            self.assertEqual(file.read(), b"abcdef")
        self.assertEqual((session.gaps, session.bytes, session.fps), (1, 6, 60))


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
class StreamDecoderTests(unittest.TestCase):
    def test_decodes_a_recording_while_it_is_written(self):
        folder = tempfile.mkdtemp()
        data = fragmented_mp4(os.path.join(folder, "source.mp4"))
        growing = os.path.join(folder, "stream.mp4")
        open(growing, "wb").close()
        done = threading.Event()
        frames = []
        decoder = StreamDecoder(growing, 30, lambda index, yuv: frames.append((index, yuv.shape)), done.is_set)
        decoder.start()
        for piece in pieces(data, 5):
            with open(growing, "ab") as file:
                file.write(piece)
            time.sleep(0.1)
        done.set()
        decoder.join(10)
        self.assertEqual(decoder.size, (320, 240))
        self.assertAlmostEqual(len(frames), 60, delta=1)
        self.assertEqual(frames[0], (1, (360, 320)))  # I420: 240 rows of Y + 120 of U and V


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
class CaptureServerTests(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.app = CaptureApp(self.out, "secret")
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()

    def serve(self, phone):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app, phone=phone))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def post(self, url, body, key=None):
        request = urllib.request.Request(url, data=body, method="POST")
        if key:
            request.add_header("X-Key", key)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def test_phone_uploads_a_recording(self):
        base = self.serve(phone=True)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post(base + "/api/start", b"{}", key="wrong")
        self.assertEqual(caught.exception.code, 403)

        info = json.dumps({"camera": "Back Camera", "width": 320, "height": 240, "fps": 30, "mime": "video/mp4"})
        session = self.post(base + "/api/start", info.encode(), key="secret")["session"]
        data = fragmented_mp4(os.path.join(self.out, "source.mp4"))
        for seq, piece in enumerate(pieces(data, 4)):
            self.assertTrue(self.post(f"{base}/api/piece?s={session}&n={seq}", piece, key="secret")["saved"])
        self.assertFalse(self.post(f"{base}/api/piece?s={session}&n=0", b"x", key="secret")["saved"])
        self.post(f"{base}/api/stop?s={session}", b"{}", key="secret")

        deadline = time.time() + 15
        while time.time() < deadline and not (self.app.session.saved_path and not self.app.decoder.is_alive()):
            time.sleep(0.1)
        status = self.app.status()["session"]
        self.assertEqual(status["saved_path"], os.path.join(self.out, session, "recording.mp4"))
        self.assertIsNone(status["error"])
        self.assertAlmostEqual(status["frames"], 60, delta=1)
        self.assertEqual(status["size"], (320, 240))
        self.assertIsNotNone(self.app.preview)
        self.assertEqual(self.app.recordings()[0]["id"], session)

    def test_status_page_on_this_computer_needs_no_key(self):
        base = self.serve(phone=False)
        with urllib.request.urlopen(base + "/api/status", timeout=10) as response:
            self.assertFalse(json.load(response)["recording"])
        with urllib.request.urlopen(base + "/camera", timeout=10) as response:
            self.assertIn(b"Badminton camera", response.read())


@unittest.skipUnless(shutil.which("openssl"), "needs openssl")
class CertificateTests(unittest.TestCase):
    def test_certificate_lists_every_address_and_is_reused(self):
        folder = tempfile.mkdtemp()
        cert, _ = ensure_certificate(folder, ["192.168.1.20", "studio.local"])
        names = subprocess.run(["openssl", "x509", "-in", cert, "-noout", "-ext", "subjectAltName"],
                               capture_output=True, text=True, check=True).stdout
        self.assertIn("IP Address:192.168.1.20", names)
        self.assertIn("DNS:studio.local", names)
        before = os.path.getmtime(cert)
        ensure_certificate(folder, ["studio.local"])
        self.assertEqual(os.path.getmtime(cert), before)


if __name__ == "__main__":
    unittest.main()

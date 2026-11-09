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

from types import SimpleNamespace

import numpy as np

from badminton_analysis.analysis.line_calls import LineCallConfig, RallyCall, call_landing
from badminton_analysis.analysis.multi_camera import COMBINED_CALLS, turn
from badminton_analysis.live.capture import (CaptureApp, CaptureSession, Phone, StreamDecoder, ensure_certificate,
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
        phone = self.app.phone()
        while time.time() < deadline and not (phone.session.saved_path and not phone.decoder.is_alive()):
            time.sleep(0.1)
        status = self.app.status()["session"]
        self.assertEqual(status["saved_path"], os.path.join(self.out, session, "recording.mp4"))
        self.assertIsNone(status["error"])
        self.assertAlmostEqual(status["frames"], 60, delta=1)
        self.assertEqual(status["size"], (320, 240))
        self.assertIsNotNone(self.app.preview())
        self.assertEqual(self.app.recordings()[0]["id"], session)
        self.assertEqual(self.app.recordings()[0]["phone"], "Phone 1")

    def test_court_is_marked_on_this_computer_and_kept(self):
        phone, local = self.serve(phone=True), self.serve(phone=False)
        with self.assertRaises(urllib.error.HTTPError) as caught:   # no picture to mark it on yet
            self.post(local + "/api/court/suggest", b"")
        self.assertEqual(caught.exception.code, 404)
        self.app.phones["a"] = Phone("a", 1)
        self.app.phones["a"].frame = np.full((240, 320, 3), 90, np.uint8)
        self.assertEqual(self.post(local + "/api/court/suggest?phone=a", b"")["size"], [320, 240])
        corners = [[100, 80], [220, 80], [260, 200], [60, 200]]
        with self.assertRaises(urllib.error.HTTPError) as caught:   # not from the phone
            self.post(phone + "/api/court?phone=a", json.dumps({"corners": corners}).encode(), key="secret")
        self.assertEqual(caught.exception.code, 404)
        reply = self.post(local + "/api/court?phone=a", json.dumps({"corners": corners}).encode())
        self.assertEqual(reply["court"]["corners"], corners)
        with open(os.path.join(self.out, "court.json"), encoding="utf-8") as file:
            self.assertEqual(json.load(file)["corners"], corners)
        self.assertEqual(self.app.status()["court"]["corners"], corners)
        self.assertEqual(self.app.status()["phones"][0]["court"]["corners"], corners)

    def test_phones_film_at_once_and_one_starting_again_replaces_its_own_recording(self):
        info = {"fps": 30, "mime": "video/mp4"}
        first = self.app.start({**info, "phone": "a"})
        second = self.app.start({**info, "phone": "b"})
        self.assertEqual([(item["name"], item["recording"]) for item in self.app.status()["phones"]],
                         [("Phone 1", True), ("Phone 2", True)])
        again = self.app.start({**info, "phone": "a"})
        self.assertEqual(self.app.status(again)["session"]["id"], again)
        self.assertEqual(self.app.status(second)["session"]["info"]["phone_name"], "Phone 2")
        self.assertEqual([item["recording"] for item in self.app.status()["phones"]], [True, True])
        with self.assertRaises(LookupError):   # phone a's first recording was stopped when it started again
            self.app.add_piece(first, 0, b"x")
        for session in (second, again):
            self.app.stop(session)

    def test_calls_of_phones_filming_together_are_combined(self):
        # Phone 1 sees the landing near its far baseline 10 cm out, give or take 40; phone 2 at the other end,
        # started 3.2 s later with its court marked from its own end, sees it 7 cm in.
        live = SimpleNamespace(caller=SimpleNamespace(config=LineCallConfig()), state=lambda: None)
        seen = [(1000.0, 600, [2.0, -0.10], [0.05, 0.40]), (1003.2, 504, turn([2.0, 0.07], (True, True)), [0.05, 0.05])]
        for number, (started, frame, point, error) in enumerate(seen, 1):
            folder = os.path.join(self.out, f"phone{number}")
            os.makedirs(folder)
            phone = self.app.phones[str(number)] = Phone(str(number), number)
            phone.session = CaptureSession(folder, {"fps": 30})
            phone.session.started, phone.session.live = started, live
            call, margin = call_landing(point, "singles")
            phone.session.calls = [RallyCall(rally=1, start_frame=frame - 150, end_frame=frame, start_sec=frame / 30 - 5,
                                             end_sec=frame / 30, landing_frame=frame, landing_court=point, call=call,
                                             margin_m=margin, landing_error_m=error, landing_method="settled")]
        self.app._combine()
        combined = self.app.status()["combined"]
        self.assertEqual(combined["phones"][1]["turn"], "turned around")
        self.assertEqual(len(combined["calls"]), 1)
        self.assertEqual(combined["calls"][0]["call"], "in")
        self.assertEqual([each["call"] for each in combined["calls"][0]["phones"]], ["out", "in"])
        self.assertTrue(os.path.exists(os.path.join(self.out, "phone1", COMBINED_CALLS)))

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

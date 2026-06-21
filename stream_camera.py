#!/usr/bin/env python3
"""
Live MJPEG stream server for a Raspberry Pi CSI camera (Picamera2 / libcamera).

Tested target: Raspberry Pi 3 + Raspberry Pi OS (Bullseye/Bookworm) with
the libcamera camera stack (Picamera2).

Usage:
    python3 stream_camera.py

Then open in a browser on any device on the same network:
    http://<raspberry-pi-ip-address>:8000/
"""

import io
import logging
import socketserver
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# --- Settings you may want to tweak ---
RESOLUTION = (640, 480)   # Lower this (e.g. 480x320) if FPS is too low on a Pi 3
PORT = 8000

PAGE = """\
<html>
<head><title>Raspberry Pi Live Stream</title></head>
<body style="margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh;">
  <img src="stream.mjpg" style="max-width:100%;height:auto;" />
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    """Receives encoded JPEG frames from Picamera2 and makes the latest one
    available to any number of waiting HTTP clients."""

    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            content = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
            )
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                logging.info("Client %s disconnected", self.client_address)

        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Quieter console output; comment out to see full request logs.
        pass


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    global output

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": RESOLUTION})
    picam2.configure(config)

    output = StreamingOutput()
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))

    try:
        address = ("", PORT)
        srv = StreamingServer(address, StreamingHandler)
        print(f"Streaming started. Open http://<your-pi-ip>:{PORT}/ in a browser.")
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        picam2.stop_recording()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import io
import logging
import socketserver
from http import server
from threading import Condition, Thread
import os
import time
import RPi.GPIO as gp

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# --- Settings ---
RESOLUTION = (640, 480)
PORT = 8000
# Type ANY combination of cameras here: ["A", "B"], ["B", "D"], or ["A", "B", "C", "D"]
CAMERAS_TO_CYCLE = ["B", "C", "D"]
CYCLE_TIME = 5 # Seconds to stream each camera

# 1. DYNAMIC HTML GENERATOR
def generate_dashboard_html(cameras):
    html = """\
    <html>
    <head>
        <title>Arducam Dynamic Dashboard</title>
        <style>
            body { margin:0; background:#121212; color: #fff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            h2 { text-align: center; margin-top: 20px; letter-spacing: 2px; }
            .dashboard { display: flex; justify-content: center; gap: 40px; margin-top: 30px; padding: 0 20px; flex-wrap: wrap; }
            .cam-container { background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            .cam-container h3 { margin-top: 0; color: #4CAF50; }
            img { width: 100%; max-width: 600px; height: auto; border-radius: 4px; background: #000; }
        </style>
    </head>
    <body>
      <h2>SECURITY DASHBOARD - AUTO CYCLING</h2>
      <div class="dashboard">
    """
    
    for cam in cameras:
        html += f"""
          <div class="cam-container">
              <h3>PORT {cam}</h3>
              <img src="stream_{cam}.mjpg" alt="Camera {cam} Feed" />
          </div>
        """
        
    html += """
      </div>
    </body>
    </html>
    """
    return html

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

# Create a separate memory buffer for every camera in our list
outputs = {cam: StreamingOutput() for cam in CAMERAS_TO_CYCLE}

class StreamingHandler(server.BaseHTTPRequestHandler):
    def serve_mjpeg_stream(self, camera_id):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        
        out_buffer = outputs[camera_id]
        
        try:
            while True:
                with out_buffer.condition:
                    out_buffer.condition.wait()
                    frame = out_buffer.frame
                
                if frame is None:
                    continue
                    
                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            logging.info(f"Client disconnected from Camera {camera_id}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            content = generate_dashboard_html(CAMERAS_TO_CYCLE).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif self.path.startswith("/stream_") and self.path.endswith(".mjpg"):
            cam_id = self.path.replace("/stream_", "").replace(".mjpg", "")
            if cam_id in CAMERAS_TO_CYCLE:
                self.serve_mjpeg_stream(cam_id)
            else:
                self.send_error(404)
                self.end_headers()
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def setup_hardware(camera_port):
    cam_configs = {
        "A": {"i2c": "0x04", "pins": [False, False, True]},
        "B": {"i2c": "0x05", "pins": [True, False, True]},
        "C": {"i2c": "0x06", "pins": [False, True, False]},
        "D": {"i2c": "0x07", "pins": [True, True, False]}
    }
    
    config = cam_configs.get(camera_port.upper(), cam_configs["A"])

    gp.setwarnings(False)
    gp.setmode(gp.BOARD)
    gp.setup(7, gp.OUT)
    gp.setup(11, gp.OUT)
    gp.setup(12, gp.OUT)
    
    # Send I2C command to hidden Pi 5 Bus 10
    os.system(f"sudo i2cset -y 10 0x70 0x00 {config['i2c']}")
    gp.output(7, config["pins"][0])
    gp.output(11, config["pins"][1])
    gp.output(12, config["pins"][2])
    
    # WARMUP DELAY: Allow electrical lanes and sensor to stabilize
    time.sleep(1.5)

def camera_cycle_worker():
    idx = 0
    while True:
        active_cam = CAMERAS_TO_CYCLE[idx]
        print(f"--- Activating Camera {active_cam} ---")
        
        setup_hardware(active_cam)
        active_buffer = outputs[active_cam]
        
        try:
            picam2 = Picamera2()
            config = picam2.create_video_configuration(main={"size": RESOLUTION})
            picam2.configure(config)
            picam2.start_recording(MJPEGEncoder(), FileOutput(active_buffer))
            
            time.sleep(CYCLE_TIME)
            
        except Exception as e:
            print(f"Camera Error: {e}")
            time.sleep(2)
            
        finally:
            try:
                picam2.stop_recording()
                picam2.close()
            except:
                pass
            
            # COOLDOWN DELAY: Wait for kernel to release /dev/video0 completely
            time.sleep(1.0)
        
        idx = (idx + 1) % len(CAMERAS_TO_CYCLE)

def main():
    cycler_thread = Thread(target=camera_cycle_worker, daemon=True)
    cycler_thread.start()

    try:
        address = ("", PORT)
        srv = StreamingServer(address, StreamingHandler)
        print(f"Dynamic Dashboard Server started. Open http://<your-pi-ip>:{PORT}/ in a browser.")
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        gp.cleanup()

if __name__ == "__main__":
    main()
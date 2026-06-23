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

# Tracks which camera is currently active — updated by camera_cycle_worker
active_camera = None

# 1. DYNAMIC HTML GENERATOR
def generate_dashboard_html(cameras):
    cam_ids = ", ".join(f'"{c}"' for c in cameras)
    html = f"""\
    <html>
    <head>
        <title>Traffic Camera Dashboard</title>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ background: #0d0d0d; color: #fff; font-family: 'Segoe UI', sans-serif; }}

            header {{
                display: flex; align-items: center; justify-content: center;
                gap: 12px; padding: 18px 0 10px;
            }}
            header h2 {{ font-size: 1.3rem; letter-spacing: 3px; color: #e0e0e0; }}
            .dot {{ width: 10px; height: 10px; border-radius: 50%; background: #4CAF50;
                    box-shadow: 0 0 8px #4CAF50; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%,100%{{ opacity:1 }} 50%{{ opacity:0.3 }} }}

            .dashboard {{
                display: flex; justify-content: center; gap: 28px;
                padding: 20px; flex-wrap: wrap;
            }}

            .cam-card {{
                background: #1a1a1a;
                border: 2px solid #333;
                border-radius: 10px;
                padding: 14px;
                text-align: center;
                width: 340px;
                transition: border-color 0.3s, box-shadow 0.3s, opacity 0.3s;
                opacity: 0.45;
            }}
            .cam-card.active {{
                border-color: #4CAF50;
                box-shadow: 0 0 18px rgba(76,175,80,0.55);
                opacity: 1;
            }}

            .cam-header {{
                display: flex; align-items: center; justify-content: center;
                gap: 10px; margin-bottom: 10px;
            }}
            .cam-header h3 {{ font-size: 1rem; letter-spacing: 1px; color: #ccc; }}
            .cam-card.active .cam-header h3 {{ color: #4CAF50; }}

            .live-badge {{
                display: none; align-items: center; gap: 5px;
                background: #c0392b; color: #fff;
                font-size: 0.68rem; font-weight: 700; letter-spacing: 1px;
                padding: 2px 8px; border-radius: 4px;
            }}
            .live-badge .rec {{ width: 7px; height: 7px; border-radius: 50%;
                                background: #fff; animation: pulse 0.9s infinite; }}
            .cam-card.active .live-badge {{ display: flex; }}

            .status-label {{
                font-size: 0.7rem; color: #555; letter-spacing: 1px;
                margin-bottom: 8px; text-transform: uppercase;
            }}
            .cam-card.active .status-label {{ color: #4CAF50; }}

            img {{
                width: 100%; border-radius: 6px; background: #000;
                display: block;
            }}
        </style>
    </head>
    <body>
      <header>
        <div class="dot"></div>
        <h2>TRAFFIC CAMERA DASHBOARD</h2>
        <div class="dot"></div>
      </header>

      <div class="dashboard">
    """

    for cam in cameras:
        html += f"""
        <div class="cam-card" id="card-{cam}">
          <div class="cam-header">
            <h3>LANE {cam}</h3>
            <div class="live-badge"><div class="rec"></div> LIVE</div>
          </div>
          <div class="status-label" id="status-{cam}">STANDBY</div>
          <img src="stream_{cam}.mjpg" alt="Lane {cam}" />
        </div>
        """

    html += f"""
      </div>

      <script>
        const cameras = [{cam_ids}];
        let currentActive = null;

        function updateActive(cam) {{
          if (cam === currentActive) return;
          cameras.forEach(c => {{
            const card   = document.getElementById('card-' + c);
            const status = document.getElementById('status-' + c);
            if (c === cam) {{
              card.classList.add('active');
              status.textContent = 'LIVE';
            }} else {{
              card.classList.remove('active');
              status.textContent = 'STANDBY';
            }}
          }});
          currentActive = cam;
        }}

        async function poll() {{
          try {{
            const r = await fetch('/active_cam');
            const d = await r.json();
            if (d.cam) updateActive(d.cam);
          }} catch(e) {{}}
        }}

        poll();
        setInterval(poll, 500);
      </script>
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

        elif self.path == "/active_cam":
            import json
            body = json.dumps({"cam": active_camera}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
    global active_camera
    idx = 0
    while True:
        active_cam = CAMERAS_TO_CYCLE[idx]
        print(f"--- Activating Camera {active_cam} ---")
        active_camera = active_cam

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
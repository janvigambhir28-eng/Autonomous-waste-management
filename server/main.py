
"""Mock telemetry backend for the waste tracking robot.
 
Run:
    pip install -r ../requirements.txt
    python -m uvicorn main:app
    then open http://localhost:8000
 
Endpoints:
    GET  /api/video          live camera stream with detection boxes
    GET  /api/telemetry      latest snapshot (poll this)
    WS   /ws/telemetry       same snapshot pushed every second
    POST /api/control        override status, emergency stop, reset route
"""
import asyncio
import math
import random
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional
 
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import sys
import threading
import time
from pathlib import Path
 
import cv2
import numpy as np
 
# Make the sibling vision_module folder importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vision_module.detect_waste import detect, draw  # noqa: E402
 
Status = Literal["Scanning", "Navigating", "Hazard Detected"]
Command = Literal["override", "resume_auto", "emergency_stop", "release_stop", "reset_route"]
 
BOT_ID = "sweeper-07"
WAYPOINTS = [(0, 0), (24, 0), (24, 14), (6, 14), (6, 6), (0, 6)]
SPEED_MPS = 0.8
TICK_S = 1.0
 
 
class Coordinates(BaseModel):
    x: float
    y: float
 
 
class LogEvent(BaseModel):
    timestamp: str
    level: Literal["info", "warning", "critical"]
    message: str
 
 
class Telemetry(BaseModel):
    bot_id: str
    battery: float
    status: Status
    coordinates: Coordinates
    log_events: list[LogEvent]
    mode: Literal["Auto", "Manual"]
    emergency_stop: bool
    timestamp: str
 
 
class ControlRequest(BaseModel):
    command: Command
    status: Optional[Status] = None  # only used with "override"
 
 
class Bot:
    def __init__(self) -> None:
        self.battery = 82.0
        self.status: Status = "Navigating"
        self.mode = "Auto"
        self.stopped = False
        self.x, self.y = 0.0, 0.0
        self.wp = 1
        self.hold = 0.0  # seconds left in Scanning / Hazard Detected
        self.vision_hazard = False  # set by the camera thread
        self.events: deque[LogEvent] = deque(maxlen=50)
        self.log("info", "Route started in auto mode")
 
    def log(self, level: str, message: str) -> None:
        self.events.append(LogEvent(timestamp=_now(), level=level, message=message))
 
    def set_status(self, status: Status, hold: float = 0.0) -> None:
        if status != self.status:
            level = "warning" if status == "Hazard Detected" else "info"
            self.log(level, f"Status changed to {status}")
        self.status = status
        self.hold = hold
 
    def tick(self, dt: float) -> None:
        if self.stopped:
            return
        self.battery = max(0.0, self.battery - 0.02 * dt)
 
        if self.mode == "Auto":
            if self.vision_hazard:
                self.set_status("Hazard Detected", hold=2)  # camera sees a hazard
            elif self.status == "Navigating":
                if random.random() < 0.10 * dt:
                    self.set_status("Scanning", hold=3)
            else:
                self.hold -= dt
                if self.hold <= 0:
                    self.set_status("Navigating")
 
        if self.status == "Navigating":
            self._move(dt)
 
    def _move(self, dt: float) -> None:
        tx, ty = WAYPOINTS[self.wp]
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        step = SPEED_MPS * dt
        if dist <= step:
            self.x, self.y = float(tx), float(ty)
            self.wp = (self.wp + 1) % len(WAYPOINTS)
        else:
            self.x += dx / dist * step
            self.y += dy / dist * step
 
    def snapshot(self) -> Telemetry:
        return Telemetry(
            bot_id=BOT_ID,
            battery=round(self.battery, 1),
            status=self.status,
            coordinates=Coordinates(x=round(self.x, 2), y=round(self.y, 2)),
            log_events=list(self.events)[-10:][::-1],  # newest first
            mode=self.mode,  # type: ignore[arg-type]
            emergency_stop=self.stopped,
            timestamp=_now(),
        )
 
 
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
 
 
bot = Bot()
 
 
async def simulate() -> None:
    while True:
        await asyncio.sleep(TICK_S)
        bot.tick(TICK_S)
 
 
# ---- Camera: runs in a background thread and feeds the vision module ----
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0")  # webcam index or video path
frame_lock = threading.Lock()
stop_event = threading.Event()
latest_jpeg = b""
 
 
def placeholder(text: str) -> np.ndarray:
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(img, text, (170, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    return img
 
 
def camera_loop() -> None:
    global latest_jpeg
    is_file = not CAMERA_SOURCE.isdigit()
    cap = cv2.VideoCapture(CAMERA_SOURCE if is_file else int(CAMERA_SOURCE))
    n, fps, last = 0, 0.0, time.perf_counter()
    while not stop_event.is_set():
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok:
            if cap.isOpened() and is_file:  # video ended: play again
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            bot.vision_hazard = False
            frame = placeholder("No camera found")
            time.sleep(0.5)
        else:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
            n += 1
            detections, _ = detect(frame, 800)
            bot.vision_hazard = any(d[0] == "Hazard" for d in detections)
            if n % 10 == 0:
                for label, x, y, bw, bh in detections:
                    print(f"frame={n:05d} {label:<10} x={x} y={y} w={bw} h={bh}")
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
            last = now
            draw(frame, detections, n, fps)
            if is_file:
                time.sleep(0.03)
        done, buf = cv2.imencode(".jpg", frame)
        if done:
            with frame_lock:
                latest_jpeg = buf.tobytes()
    cap.release()
 
 
@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(simulate())
    threading.Thread(target=camera_loop, daemon=True).start()
    yield
    stop_event.set()
    task.cancel()
 
 
app = FastAPI(title="Waste Robot Telemetry", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
@app.get("/api/telemetry", response_model=Telemetry)
async def get_telemetry() -> Telemetry:
    return bot.snapshot()
 
 
@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            await ws.send_json(bot.snapshot().model_dump())
            await asyncio.sleep(TICK_S)
    except WebSocketDisconnect:
        pass
 
 
@app.post("/api/control", response_model=Telemetry)
async def control(req: ControlRequest) -> Telemetry:
    if req.status and req.command != "override":
        raise HTTPException(400, "'status' can only be set with the 'override' command")
 
    if req.command == "override":
        if bot.stopped:
            raise HTTPException(409, "Release the emergency stop before overriding")
        if bot.mode != "Manual":
            bot.mode = "Manual"
            bot.log("warning", "Operator override: manual mode")
        if req.status:
            bot.set_status(req.status)
    elif req.command == "resume_auto":
        bot.mode = "Auto"
        bot.set_status("Navigating")
        bot.log("info", "Auto mode resumed")
    elif req.command == "emergency_stop":
        bot.stopped = True
        bot.log("critical", "Emergency stop engaged")
    elif req.command == "release_stop":
        bot.stopped = False
        bot.log("info", "Emergency stop released")
    elif req.command == "reset_route":
        bot.x, bot.y, bot.wp = 0.0, 0.0, 1
        bot.log("info", "Route reset to start")
 
    return bot.snapshot()
 
 
def mjpeg():
    while True:
        with frame_lock:
            jpg = latest_jpeg
        if jpg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.04)
 
 
@app.get("/api/video")
def video() -> StreamingResponse:
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")
 
 
@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse("/dashboard/dashboard.html")
 
 
# Keep this last so it never shadows the /api routes.
app.mount("/dashboard", StaticFiles(directory=ROOT / "dashboard"), name="dashboard")
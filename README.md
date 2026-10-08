# Autonomous Waste Management

A small demo of an autonomous waste tracking robot: a camera detects waste and hazards,
a Python server shares the robot's status, and a web dashboard shows everything live.

## How the parts fit together

```
Webcam / video --> vision_module/detect_waste.py (finds Hazard / Waste Item by color)
                          |
                          v
                   server/main.py (FastAPI)
                   - /api/video      live camera stream with boxes
                   - /api/telemetry  battery, status, coordinates, log
                   - /api/control    override, emergency stop, reset route
                          |
                          v
                   dashboard/dashboard.html (shows video, telemetry, buttons)
```

## Folder structure

```
autonomous-waste-management/
  dashboard/dashboard.html
  server/main.py
  vision_module/detect_waste.py
  requirements.txt
  README.md
```

## Run it

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and install the packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   cd server
   python -m uvicorn main:app
   ```
4. Open http://localhost:8000 in your browser.

To use a video file instead of the webcam:

```bash
# macOS / Linux
CAMERA_SOURCE=clip.mp4 python -m uvicorn main:app
# Windows (PowerShell)
$env:CAMERA_SOURCE="clip.mp4"; python -m uvicorn main:app
```

If the server is not running, the dashboard falls back to a built-in demo simulation.

## Testing the camera

Hold up a red or orange object for **Hazard**, and a blue or green object for **Waste Item**.
When a hazard is seen, the robot status changes to "Hazard Detected" on the dashboard.

You can also run the vision module on its own: `python vision_module/detect_waste.py`.

## Notes

- Detection uses simple color thresholding, so it is sensitive to lighting.
  A pretrained model (for example YOLO) could replace `detect()` later.
- Telemetry values such as battery and coordinates are simulated.

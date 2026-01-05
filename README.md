# baby_monitor

Computer-vision baby monitoring prototype using a local OpenCV display (`cv2.imshow`).

## What’s in this repo
- `run_local.py` — simple runner / entrypoint
- `tracking.py` — main tracking + CV logic
- `yunet.py` — YuNet face detector wrapper
- `demoDay.avi` / `demoNight.avi` — short calibration videos (day vs night)

## Setup

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

### 2) Install dependencies
```bash
python -m pip install -r requirements.txt
```

## Run

### Run with the calibration videos
**Day video**
```bash
python run_local.py --source demoDay.avi
```

**Night video**
```bash
python run_local.py --source demoNight.avi
```

### Run with webcam
```bash
python run_local.py --source 0
```

## Controls
- Press **q** to quit the OpenCV window.

## How it works

Each frame follows the same pipeline:

1. **Read a frame** from the configured source (webcam/video).
2. Build two frame formats:
   - **BGR** frame for OpenCV / YuNet (OpenCV’s default color format).
   - **RGB** frame for MediaPipe Pose.
3. **Face state (YuNet):**
   - Run YuNet face detection on the BGR frame.
   - If no face is detected, treat the state as **DANGER** (face covered / not visible).
4. **Body position (MediaPipe Pose):**
   - Run MediaPipe Pose on the RGB frame.
   - Classify a coarse baby position (Face Up / Face Down / On It's Side / Covered) using landmark heuristics.
5. **Warning debouncing:**
   - Warnings are only “latched” after a few consecutive frames to reduce flicker/noise.
6. **Render overlays:**
   - YuNet detections are drawn on the display frame.
   - Optionally, pose landmarks can be blended on top for debugging.

## Debugging tips

### Enable/disable pose landmark overlay
In `tracking.py`, look for:
```python
show_pose_landmarks = True
```
Set it to `False` to improve performance.

### Enable debug logging
In `run_local.py`, set the logging level:
```python
logging.basicConfig(level=logging.INFO)  # or DEBUG for per-frame logs
```

## Models / assets
YuNet expects the ONNX model at:
- `DataSets/face_detection_yunet_2022mar.onnx`

The demo videos are included for calibration:
- `demoDay.avi`
- `demoNight.avi`

## Notes / Troubleshooting
- If the webcam doesn’t open, try another index: `--source 1`
- If you see a “model not found” error for YuNet, confirm the ONNX path above exists.

## Compatibility notes
This project uses the legacy `mediapipe.solutions` API and OpenCV’s YuNet (DNN/ONNX) face detector.

For reproducibility, dependencies are pinned in `requirements.txt`. Newer versions of MediaPipe/OpenCV/NumPy may introduce breaking changes.

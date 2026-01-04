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
pip install -r requirements.txt
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

## Notes / Troubleshooting

- If the webcam doesn’t open, try another index: `--source 1`
- If you see missing model errors for YuNet, ensure any required model file paths referenced in `tracking.py` exist on disk.

## Compatibility notes

This project uses the legacy `mediapipe.solutions` API and OpenCV’s YuNet (DNN/ONNX) face detector.

For reproducibility, dependencies are pinned in `requirements.txt`. Newer versions of MediaPipe/OpenCV/NumPy may introduce breaking changes.

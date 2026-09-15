# Real-Time Road Accident Detection and Automated Emergency Response System

A production-oriented, **phase-by-phase** build of a road accident
detection platform: **YOLO detection + ByteTrack multi-object tracking +
temporal motion/collision analysis + accident verification**, with
**GPS/location, FastAPI backend, database, and alerting** added as
separate, loosely-coupled layers.

> Development follows a strict phase order (see project spec). This
> repository currently implements **Phase 1 — Project Architecture and
> Environment** only. Every other module exists as a documented
> placeholder stating which phase implements it — this keeps the
> package layout stable so later phases are additive, not a rewrite.

## Project status

| Phase | Description | Status |
|---|---|---|
| 1 | Project architecture & environment | ✅ Done |
| 2 | Video input layer | ✅ Done |
| 3 | YOLO object detection | ✅ Done |
| 4 | Multi-object tracking (ByteTrack) | ⏳ Not started |
| 5 | Movement & trajectory analysis | ⏳ Not started |
| 6 | Collision detection | ⏳ Not started |
| 7 | Accident detection & temporal verification | ⏳ Not started |
| 8 | Accident event generation | ⏳ Not started |
| 9 | Evidence capture | ⏳ Not started |
| 10 | GPS & location system | ⏳ Not started |
| 11 | FastAPI backend | ⏳ Not started |
| 12 | Database layer | ⏳ Not started |
| 13 | Alert/notification service | ⏳ Not started |
| 14 | API-ready architecture hardening | ⏳ Not started |
| 15 | Testing & evaluation | ⏳ Not started |
| 16 | Performance optimization | ⏳ Not started |
| 17 | Final integration | ⏳ Not started |

## Project structure

```
road-accident-detection/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/                  # Environment-driven settings + tunable thresholds
│   ├── settings.py
│   ├── threshold_loader.py
│   └── thresholds.yaml
├── utils/
│   └── logger.py            # Shared logging setup
├── models/yolo/              # YOLO weights (not committed — see below)
├── data/
│   ├── input/{videos,streams}
│   ├── output/{annotated,processed}
│   ├── evidence/{images,videos}
│   └── database/
├── engine/                   # CV engine — usable independently of FastAPI
│   ├── pipeline/              # Phase 17 wiring
│   ├── input/                 # Phase 2
│   ├── detection/             # Phase 3
│   ├── tracking/              # Phase 4
│   ├── motion/                # Phase 5
│   ├── collision/             # Phase 6
│   ├── accident/              # Phase 7
│   ├── evidence/              # Phase 9
│   └── events/                # Phase 8
├── location/                  # Phase 10 — independent of the CV engine
├── backend/                    # Phase 11-14 — FastAPI app
│   ├── main.py
│   ├── api/v1/
│   ├── schemas/
│   ├── services/
│   ├── database/
│   └── alerts/                 # Phase 13
├── scripts/                    # CLI entry points (run_detection.py etc.)
├── tests/
└── logs/
```

No Docker files are included by design — this project targets a plain
Python virtual-environment workflow.

## Installation

Requires **Python 3.10+**.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your local environment file
cp .env.example .env
# Edit .env if you want to override any default (model path, thresholds, etc.)
```

### YOLO weights

`models/yolo/` is empty by default (weights are large binaries and are
not committed). Ultralytics will auto-download the default nano model
(`yolov8n.pt`) into that folder the first time detection runs (Phase 3)
if `YOLO_MODEL_PATH` points there and the file doesn't yet exist. No
model is downloaded or trained as part of Phase 1.

## Configuration

All configuration is environment-driven — see `.env.example` for every
supported variable and `config/settings.py` for defaults and types.
Fine-grained scoring weights (used starting Phase 6) live in
`config/thresholds.yaml`.

Nothing is hardcoded: paths, thresholds, model names, coordinates, and
API keys are all sourced from `Settings` or `thresholds.yaml`, never
inlined in application code.

## Verifying Phase 1

Run the setup check — it imports the config layer, confirms every data
directory exists (creating them if needed), and validates
`thresholds.yaml`:

```bash
python scripts/check_setup.py
```

Or run the automated test:

```bash
pytest tests/test_setup.py -v
```

Expected output: all checks pass, and a line confirming the settings
and threshold config loaded successfully.

## Verifying Phase 2 — Video Input Layer

Phase 2 adds a `VideoSource` interface (`engine/input/video_source.py`)
with three implementations — `FileVideoSource`, `WebcamVideoSource`,
`RTSPVideoSource` — selected at runtime via `create_video_source()`
based on `VIDEO_SOURCE_TYPE` in `.env`. All three share:

- **FPS handling** — reported via `get_fps()`, `0.0` when unknown
  (e.g. many webcams).
- **Frame dimensions** — via `get_frame_size()`.
- **Stream failure handling** — `read()` returns a `FrameResult` with
  `success=False` on failure instead of raising; `RTSPVideoSource`
  additionally retries a configurable number of times
  (`RTSP_RECONNECT_ATTEMPTS`, `RTSP_RECONNECT_DELAY_SEC`) after
  `RTSP_READ_FAILURE_LIMIT` consecutive failed reads, since network
  camera streams are the most failure-prone input.
- **Graceful shutdown** — `release()` is always safe to call, including
  before a successful `open()`; every source also works as a context
  manager (`with create_video_source(settings) as source:`).
- **Optional frame skipping** — set `FRAME_SKIP` (e.g. `1` = return
  every 2nd frame) to trade temporal resolution for throughput on
  slower hardware.

Set `VIDEO_SOURCE_TYPE` / `VIDEO_SOURCE_PATH` (or `WEBCAM_INDEX`) in
`.env`, then run:

```bash
python scripts/check_video_input.py --max-frames 60
```

Or run the automated tests (self-contained — generates a small
synthetic `.mp4` on the fly, no sample footage required):

```bash
pytest tests/test_video_input.py -v
```

Expected output: the source opens, reports FPS/frame size, reads the
requested number of frames, and shuts down cleanly.

**Note:** webcam and live RTSP streams can't be exercised in an
automated/headless test environment, so `test_video_input.py` verifies
`WebcamVideoSource`/`RTSPVideoSource` via the factory and their
interface/error-handling behavior (e.g. RTSP credential masking in
logs), while full read/frame-skip behavior is tested thoroughly against
`FileVideoSource`, which all three share the same base-class logic
with. Test on a real webcam or RTSP URL directly via
`scripts/check_video_input.py` when that hardware/stream is available.

## Verifying Phase 3 — YOLO Object Detection

Phase 3 adds `engine/detection/yolo_detector.py`: a thin, config-driven
wrapper (`YOLODetector`) around a pretrained Ultralytics YOLO model.

- **Inference** — `detector.detect(frame)` runs the model on a single
  BGR frame and returns a list of structured `Detection` objects
  (`class_name`, `class_id`, `confidence`, `bbox`) — never raw
  Ultralytics result objects, so tracking/motion/collision (later
  phases) only ever depend on this module's types.
- **Filtering** — results are filtered both by `YOLO_CONFIDENCE_THRESHOLD`
  and by `YOLO_TARGET_CLASSES` (car, motorcycle, bus, truck, bicycle,
  person by default); anything else the model detects (dog, traffic
  light, etc.) is discarded.
- **Visualization** — `YOLODetector.draw_detections(frame, detections)`
  returns an annotated **copy** of the frame with bounding boxes, class
  labels, and confidence scores drawn (original frame is never
  mutated).
- **No training** — a pretrained COCO model (`yolov8n.pt`) is used as-is,
  per the project spec.

### Model weights

`YOLO_MODEL_PATH` defaults to `models/yolo/yolov8n.pt`. That file is
gitignored (binary weights aren't committed). The first time you load
the detector, Ultralytics automatically downloads the nano model to
that exact path if it isn't already there — no manual step needed, as
long as the environment has internet access. Swap in a different
Ultralytics-supported model by changing `YOLO_MODEL_PATH`.

### Running it

```bash
# in .env: VIDEO_SOURCE_TYPE=file, VIDEO_SOURCE_PATH=path/to/your/video.mp4
python scripts/run_detection.py --max-frames 100 --output my_run.mp4
```

This reads frames via the Phase 2 video input layer, runs detection on
each one, logs every frame's detections as structured JSON-like dicts,
and writes an annotated video to `data/output/annotated/`.

```bash
pytest tests/test_detection.py -v
```

Expected output: dataclass/error-handling tests always run; the
model-dependent tests load the real pretrained model (downloading it
if needed) and assert against `tests/fixtures/sample_scene.jpg` — a
real photo containing a bus and several pedestrians — confirming
actual detections, correct class/confidence filtering, and bbox
sanity. These tests `skip` (not fail) if the model genuinely can't be
loaded in a given environment (e.g. fully offline).

## Module overview (Phase 1)

- **`config/settings.py`** — the single `Settings` object (pydantic
  `BaseSettings`) that every other layer imports. Reads from `.env`,
  falls back to sane defaults, and creates required directories on
  import.
- **`config/thresholds.yaml`** + **`config/threshold_loader.py`** —
  detailed, hand-tunable scoring weights for collision/accident logic,
  kept separate from `settings.py` so they can be edited without
  touching environment variables.
- **`utils/logger.py`** — one shared logger factory (`get_logger(name)`)
  used by every module; writes to both console and a rotating file
  under `logs/`.
- **`engine/`** — the computer-vision engine, split into one
  sub-package per pipeline stage (input → detection → tracking →
  motion → collision → accident → evidence → events). Each stage is
  independent so it can be tested and modified without touching the
  others.
  - **`engine/input/`** (Phase 2, implemented) — `VideoSource`
    interface + `FileVideoSource` / `WebcamVideoSource` /
    `RTSPVideoSource`, selected via `create_video_source(settings)`.
  - **`engine/detection/`** (Phase 3, implemented) — `YOLODetector`
    wrapping a pretrained Ultralytics model; `Detection`/`BoundingBox`
    dataclasses; `draw_detections()` for visualization.
  - All other `engine/` sub-packages are documented placeholders; real
    logic starts Phase 4.
- **`location/`** — GPS/fixed-camera location logic, deliberately
  decoupled from `engine/` (per requirement: "Keep GPS independent from
  the CV engine"). Implemented Phase 10.
- **`backend/`** — the FastAPI application. Routes (`api/v1/`) contain
  no business logic; that lives in `services/`, which in turn uses
  `database/repositories/` for persistence. Implemented Phase 11-14.
- **`scripts/`** — CLI entry points such as `run_detection.py`,
  `run_webcam.py`, `run_stream.py`, `test_alert.py` (added as their
  underlying phases are implemented) and `check_setup.py` (Phase 1).
- **`tests/`** — one test module per concern (`test_setup.py` for
  Phase 1; `test_detection.py`, `test_tracking.py`, etc. added phase by
  phase).

## Design notes

- The CV engine (`engine/`) never imports from `backend/`, so it can be
  run and tested completely standalone (e.g. via `scripts/`) without a
  running API server.
- `location/` never imports from `engine/`, so GPS/camera-location
  logic can be swapped or extended (e.g. a real GPS device) without
  touching detection/tracking code.
- Alert providers (Phase 13) will implement a common interface so a new
  destination (e.g. an authorized emergency-service API) can be added
  without changing the alert-dispatch logic.
- This is a decision-support system. It does not contact real
  emergency services and does not claim to derive real-world vehicle
  speed from uncalibrated CCTV footage.

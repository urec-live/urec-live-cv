# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`urec-live-cv` is the computer-vision microservice for the UREC Live fitness app. It detects gym equipment occupancy from video feeds or a live webcam using YOLO person detection and zone-based presence logic, then exposes that state via a FastAPI server.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the API server (port 8000 by default, auto-reloads in dev)
python scripts/run_api.py

# Run the CV pipeline demo against videos
python scripts/run_video_demo.py                             # all videos in gym_videos/
python scripts/run_video_demo.py --video gym_videos/foo.mp4  # specific file
python scripts/run_video_demo.py --video gym_videos --zones configs/zones.json
python scripts/run_video_demo.py --video gym_videos --zones-map configs/zones_map.json

# Post pipeline output directly to the running API
python scripts/run_video_demo.py --post-url http://127.0.0.1:8000/equipment/status

# Live webcam pipeline
python scripts/run_camera_demo.py --calibrate                          # discover zone pixel coords
python scripts/run_camera_demo.py --post-url http://127.0.0.1:8000/equipment/status
python scripts/run_camera_demo.py \
  --post-url http://127.0.0.1:8000/equipment/status \
  --springboot-url http://172.20.1.229:8080                            # also writes to DB immediately
```

There are no automated tests in this repo yet.

## Architecture

### Two independent processes

1. **FastAPI server** (`src/backend/main.py`) — stateless HTTP API with an in-memory status store. Receives status updates and serves them to the UREC Live backend.

2. **CV pipeline** (`scripts/run_video_demo.py` or `scripts/run_camera_demo.py`) — reads video files or webcam frames, runs YOLO inference per frame, and either prints JSON to stdout or POSTs to the API.

These communicate only via HTTP; the pipeline is not imported by the server.

### CV pipeline data flow

```
video/camera frames
  → _iter_video_frames / _iter_camera_frames (cv2, sampled every frame_interval_sec)
  → YOLO predict (yolov8n.pt for camera, yolov8x.pt for video; COCO class 0 = person)
  → zone occupancy check (person bbox center inside zone rect)
  → update_equipment_state (persist/cooldown hysteresis)
  → yield EquipmentStatus (on status change only)
```

Key timing parameters (set in zones config JSON):
- `in_use_persist_sec` — person must be in zone this long before `IN_USE` (default 3s video, 30s camera)
- `available_cooldown_sec` — grace period before reverting to `AVAILABLE` (default 2s video, 60s camera)

### Module layout (`src/`)

- `gymcv/zones.py` — `Zone` dataclass + `load_zones_from_config`
- `gymcv/usage.py` — `EquipmentState`, `EquipmentStatus`, `update_equipment_state`, `build_status_payload`
- `gymcv/pipeline.py` — `PipelineConfig`, `load_pipeline_config`, `run_usage_pipeline_on_video`, `run_usage_pipeline_on_camera`
- `backend/main.py` — FastAPI app with in-memory `STATUS_STORE`; `GET/POST /equipment/status`, `GET /health`

### Zones configuration

- `configs/zones.json` (copy from `zones.example.json`) — for video pipeline; maps equipment to pixel bounding boxes `[x1, y1, x2, y2]`
- `configs/laptop_zones.json` — for webcam pipeline; uses equipment codes `BP001` and `IP001` matching the Spring Boot `equipment.code` column
- For multi-camera setups use `zones_map.json` to route video paths to different zone configs
- Run `python scripts/run_camera_demo.py --calibrate` to discover pixel coordinates for `laptop_zones.json`

### Integration with UREC Live backend

See `docs/db-update-flow.md` for the full diagram. In short:

| Mode | FastAPI updated | DB written | Broadcast |
|---|---|---|---|
| `--post-url` only | Yes | No (polling lag) | Yes (after next poll) |
| `--post-url` + `--springboot-url` | Yes | Yes (immediate) | Yes (immediate) |
| `--springboot-url` only | No | Yes (immediate) | Yes (immediate) |

Spring Boot config (polling mode):
```properties
APP_CV_SECONDARY_ENABLED=true
APP_CV_SECONDARY_BASE_URL=http://localhost:8000
APP_CV_SECONDARY_CONFIDENCE_THRESHOLD=0.65
APP_CV_SECONDARY_CACHE_TTL_MS=3000
```

### Deployment

Deployed on Railway via Docker. `scripts/run_api.py` is the entrypoint. `PORT` env var overrides port 8000. `ENVIRONMENT=development` enables uvicorn auto-reload.

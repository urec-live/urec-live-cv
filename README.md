# Smart Gym Equipment Usage Detection (YOLO + Zones)

This repo is an end-to-end MVP for detecting **whether gym equipment is actively in use** from video feeds, using:

- **YOLO (Ultralytics)** for person detection
- **Zone occupancy** (per equipment ROI)
- **Presence persistence** over time to decide *machine in use*
- **Persistence logic**: in-zone presence for \(>3s\) → `IN_USE`
- **FastAPI** backend to publish / receive equipment status updates

It is designed for CCTV/RTSP in production, but supports **prerecorded `.mp4` videos** during development.

## Folder layout

- `gym_videos/`: development videos (not committed)
- `configs/zones.example.json`: example equipment zones and settings
- `src/gymcv/`: computer-vision pipeline modules
- `src/backend/`: FastAPI server
- `scripts/`: runnable entrypoints

## Setup

Create a virtualenv, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Add videos

Place videos here:

```
gym_videos/
  treadmill1.mp4
  bench_press1.mp4
```

## Configure zones

Copy the example and adjust coordinates for your camera view:

```bash
copy configs\\zones.example.json configs\\zones.json
```

Zones are rectangles: `[x1, y1, x2, y2]` in pixel coordinates.

## Run the API

```bash
python scripts/run_api.py
```

API:
- `GET /health`
- `GET /equipment/status`
- `POST /equipment/status` (accepts a single status update)

## Run the video pipeline demo

In another terminal:

```bash
# Process all videos under gym_videos (default behavior)
python scripts/run_video_demo.py
```

Or a specific file/folder/prefix:

```bash
python scripts/run_video_demo.py --video gym_videos\\gymdataset1
python scripts/run_video_demo.py --video gym_videos\\gymdataset1\\test\\test\\push-up
python scripts/run_video_demo.py --video gym_videos\\gymdataset1\\test\\test\\push-up.mp4
```

Use explicit zones config (recommended):

```bash
python scripts/run_video_demo.py --video gym_videos --zones configs\\zones.json
```

For multi-camera / multi-view setups, route different folders to different zone configs:

```bash
copy configs\\zones_map.example.json configs\\zones_map.json
python scripts/run_video_demo.py --video gym_videos --zones-map configs\\zones_map.json
```

The demo will:
- process frames like a live stream (default: every 0.5s)
- compute equipment status
- print status changes
- optionally POST changes to the API

## Notes / Next steps

- For best accuracy you’ll want an equipment-trained YOLO model that reliably detects people and machine-specific zones in your camera views.
- The MVP uses only human presence inside a configured machine zone; you can later refine it with better zone calibration or multi-camera fusion.

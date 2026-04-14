from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gymcv.pipeline import load_pipeline_config, run_usage_pipeline_on_video

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}


def _post_json(url: str, data: dict, timeout_sec: float = 2.0) -> None:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        resp.read()


def _resolve_default_zones_path() -> Optional[Path]:
    p1 = ROOT / "configs" / "zones.json"
    if p1.exists():
        return p1
    p2 = ROOT / "configs" / "zones.example.json"
    if p2.exists():
        return p2
    return None


def _collect_videos_from_input(video_input: str) -> list[Path]:
    """
    Accepts:
      - a direct file path
      - a directory path (recursively scans video files)
      - a file prefix without extension (tries matching siblings and recursive parent)
    """
    p = Path(video_input)
    if p.is_file():
        return [p]

    if p.is_dir():
        vids = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in VIDEO_EXTS]
        return sorted(vids)

    # Prefix mode: e.g. gym_videos/.../push-up (no extension)
    parent = p.parent if str(p.parent) else Path(".")
    stem = p.name
    matches: list[Path] = []
    if parent.exists():
        for x in parent.rglob("*"):
            if not x.is_file():
                continue
            if x.suffix.lower() not in VIDEO_EXTS:
                continue
            if x.stem == stem or x.name.startswith(stem):
                matches.append(x)
    return sorted(matches)


def _load_zones_map(path: Optional[str]) -> dict:
    if not path:
        return {"rules": [], "default": None}
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    rules = data.get("rules", []) if isinstance(data, dict) else []
    default = data.get("default") if isinstance(data, dict) else None
    return {"rules": rules, "default": default}


def _pick_zones_for_video(video: Path, zones_map: dict, fallback_zones: Optional[Path]) -> Path:
    video_str = str(video).lower()
    for rule in zones_map.get("rules", []):
        match = str(rule.get("match", "")).lower()
        zones = rule.get("zones")
        if match and zones and match in video_str:
            zp = Path(zones)
            if zp.exists():
                return zp
    default_from_map = zones_map.get("default")
    if default_from_map:
        dp = Path(default_from_map)
        if dp.exists():
            return dp
    if fallback_zones and fallback_zones.exists():
        return fallback_zones
    raise FileNotFoundError("No usable zones config found. Provide --zones and/or --zones-map.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="gym_videos",
        help="Video input path: file, directory, or name prefix without extension",
    )
    parser.add_argument("--zones", default=None, help="Default zones config path (optional if configs/zones.json exists)")
    parser.add_argument(
        "--zones-map",
        default=None,
        help="Optional JSON mapping for multi-camera views. See configs/zones_map.example.json",
    )
    parser.add_argument("--det-model", default="yolov8x.pt", help="Ultralytics detection model path/name")
    parser.add_argument("--pose-model", default="yolov8x-pose.pt", help="Ultralytics pose model path/name")
    parser.add_argument("--device", default=None, help="e.g. 'cpu', '0' for GPU")
    parser.add_argument("--post-url", default=None, help="Optional API URL, e.g. http://127.0.0.1:8000/equipment/status")
    args = parser.parse_args(argv)

    default_zones = Path(args.zones) if args.zones else _resolve_default_zones_path()
    zones_map = _load_zones_map(args.zones_map)
    videos = _collect_videos_from_input(args.video)
    if not videos:
        print(
            "No video files found for input: "
            f"{args.video}\n"
            "Tip: pass a full filename with extension, or pass a folder containing videos.",
            file=sys.stderr,
        )
        return 2

    for video in videos:
        zones_path = _pick_zones_for_video(video, zones_map, default_zones)
        cfg = load_pipeline_config(zones_path)
        print(f"Processing video: {video} | zones: {zones_path}")
        for status in run_usage_pipeline_on_video(
            video_path=video,
            cfg=cfg,
            det_model_path=args.det_model,
            pose_model_path=args.pose_model,
            device=args.device,
        ):
            payload = asdict(status)
            print(json.dumps(payload, ensure_ascii=False))
            if args.post_url:
                try:
                    _post_json(args.post_url, payload)
                except Exception as e:
                    print(f"POST failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


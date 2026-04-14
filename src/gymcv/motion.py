from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


# COCO keypoint names used by Ultralytics pose models (17 points).
COCO_KP_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_KP_INDEX: Dict[str, int] = {name: i for i, name in enumerate(COCO_KP_NAMES)}


@dataclass
class MotionConfig:
    keypoints_to_use: Tuple[str, ...] = (
        "left_wrist",
        "right_wrist",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    min_mean_pixel_movement: float = 10.0


def _select_keypoints_xy(keypoints_xy: np.ndarray, names: Iterable[str]) -> np.ndarray:
    idxs = [COCO_KP_INDEX[n] for n in names if n in COCO_KP_INDEX]
    if not idxs:
        return keypoints_xy[:0]
    return keypoints_xy[idxs]


def compute_pose_motion_score(
    prev_kps_xy: Optional[np.ndarray],
    curr_kps_xy: Optional[np.ndarray],
    cfg: MotionConfig,
) -> float:
    """
    Returns mean per-keypoint displacement in pixels.

    prev_kps_xy/curr_kps_xy: shape (17,2) arrays for a single person.
    """
    if prev_kps_xy is None or curr_kps_xy is None:
        return 0.0

    prev_sel = _select_keypoints_xy(prev_kps_xy, cfg.keypoints_to_use)
    curr_sel = _select_keypoints_xy(curr_kps_xy, cfg.keypoints_to_use)
    if prev_sel.shape != curr_sel.shape or prev_sel.size == 0:
        return 0.0

    diffs = curr_sel - prev_sel
    dists = np.sqrt((diffs**2).sum(axis=1))
    return float(np.mean(dists)) if dists.size else 0.0


def is_motion_active(motion_score: float, cfg: MotionConfig) -> bool:
    return motion_score >= float(cfg.min_mean_pixel_movement)


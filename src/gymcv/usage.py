from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class EquipmentStatus:
    equipment_id: str
    equipment_type: str
    status: str  # "IN_USE" | "AVAILABLE"
    confidence: float
    timestamp: str  # ISO8601


@dataclass
class EquipmentState:
    last_seen_person_ts: Optional[float] = None
    last_seen_motion_ts: Optional[float] = None
    in_use_since_ts: Optional[float] = None
    status: str = "AVAILABLE"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_equipment_state(
    *,
    state: EquipmentState,
    now_ts: float,
    person_in_zone: bool,
    motion_active: bool,
    persist_sec: float,
) -> EquipmentState:
    if person_in_zone:
        state.last_seen_person_ts = now_ts
    if motion_active:
        state.last_seen_motion_ts = now_ts

    # "Interaction" requires both signals close in time.
    interaction = False
    if state.last_seen_person_ts is not None and state.last_seen_motion_ts is not None:
        interaction = abs(state.last_seen_person_ts - state.last_seen_motion_ts) <= 1.0

    if interaction:
        if state.in_use_since_ts is None:
            state.in_use_since_ts = now_ts
        if (now_ts - state.in_use_since_ts) >= persist_sec:
            state.status = "IN_USE"
    else:
        state.in_use_since_ts = None
        state.status = "AVAILABLE"

    return state


def build_status_payload(
    *,
    equipment_id: str,
    equipment_type: str,
    status: str,
    confidence: float,
) -> EquipmentStatus:
    return EquipmentStatus(
        equipment_id=equipment_id,
        equipment_type=equipment_type,
        status=status,
        confidence=float(confidence),
        timestamp=_iso_now(),
    )


def status_changed(prev: Optional[EquipmentStatus], curr: EquipmentStatus) -> bool:
    if prev is None:
        return True
    return prev.status != curr.status


def init_states(equipment_ids: list[str]) -> Dict[str, EquipmentState]:
    return {eid: EquipmentState() for eid in equipment_ids}


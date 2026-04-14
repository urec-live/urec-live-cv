from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


StatusLiteral = Literal["IN_USE", "AVAILABLE"]


class EquipmentStatusIn(BaseModel):
    equipment_id: str = Field(..., examples=["bench_press_01"])
    equipment_type: Optional[str] = Field(None, examples=["bench_press"])
    status: StatusLiteral = Field(..., examples=["IN_USE"])
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    timestamp: Optional[str] = None


class EquipmentStatusOut(BaseModel):
    equipment_id: str
    equipment_type: Optional[str] = None
    status: StatusLiteral
    confidence: Optional[float] = None
    timestamp: str


app = FastAPI(title="Smart Gym Equipment Usage API", version="0.1.0")

# In-memory store for MVP (replace with DB/Redis in production).
STATUS_STORE: Dict[str, EquipmentStatusOut] = {}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/equipment/status", response_model=List[EquipmentStatusOut])
def get_all_status() -> List[EquipmentStatusOut]:
    return list(STATUS_STORE.values())


@app.post("/equipment/status", response_model=EquipmentStatusOut)
def post_status(payload: EquipmentStatusIn) -> EquipmentStatusOut:
    ts = payload.timestamp or datetime.now(timezone.utc).isoformat()
    out = EquipmentStatusOut(
        equipment_id=payload.equipment_id,
        equipment_type=payload.equipment_type,
        status=payload.status,
        confidence=payload.confidence,
        timestamp=ts,
    )
    STATUS_STORE[out.equipment_id] = out
    return out


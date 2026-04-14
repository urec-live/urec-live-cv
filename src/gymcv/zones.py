from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Zone:
    equipment_id: str
    equipment_type: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def load_zones_from_config(equipment_zones: Iterable[dict]) -> list[Zone]:
    zones: list[Zone] = []
    for z in equipment_zones:
        x1, y1, x2, y2 = z["zone_xyxy"]
        zones.append(
            Zone(
                equipment_id=str(z["equipment_id"]),
                equipment_type=str(z.get("equipment_type", "unknown")),
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
            )
        )
    return zones


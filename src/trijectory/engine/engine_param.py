import dataclasses
from typing import Literal

from trijectory.type_aliases import ArrF64


@dataclasses.dataclass
class TrajectoryParam:
    max_time: float
    time_step: float
    log_rate: int
    escape_debounce_time: float
    min_distance: float
    method: Literal["euler", "rk22", "rk44"]
    mass: ArrF64 | None

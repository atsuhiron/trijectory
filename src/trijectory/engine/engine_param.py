import dataclasses

from trijectory.type_aliases import ArrF64


@dataclasses.dataclass
class TrajectoryParam:
    max_time: float
    time_step: float
    mass: ArrF64 | None

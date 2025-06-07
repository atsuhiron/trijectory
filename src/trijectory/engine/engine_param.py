import dataclasses


@dataclasses.dataclass
class TrajectoryParam:
    max_time: float
    time_step: float

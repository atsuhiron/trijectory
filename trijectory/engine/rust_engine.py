import numpy as np

from trijectory import rs_trijectory
from trijectory.engine.base_engine import BaseEngine
from trijectory.engine.engine_param import TrajectoryParam
from trijectory.type_aliases import ArrF64


class RustEngine(BaseEngine):
    def life(self, r: ArrF64, v: ArrF64, param: TrajectoryParam) -> float:
        return rs_trijectory.calc_life(
            r,
            v,
            param.max_time,
            param.time_step,
            param.escape_debounce_time,
            param.min_distance,
            str(param.method),
            param.mass if param.mass is not None else np.ones(len(r), dtype=np.float64),
        )

    def trajectory(self, r: ArrF64, v: ArrF64, param: TrajectoryParam) -> tuple[ArrF64, ArrF64, float]:
        raise NotImplementedError

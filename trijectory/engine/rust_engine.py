import numpy as np

import rs_trijectory
from trijectory.engine.engine_param import TrajectoryParam
from trijectory.engine.base_engine import BaseEngine
from type_aliases import ArrF64


class RustEngine(BaseEngine):
    def life(self, r: ArrF64, v: ArrF64, param: TrajectoryParam) -> float:
        return rs_trijectory.calc_life(
            r,
            v,
            param.max_time,
            param.time_step,
            param.log_rate,
            param.escape_debounce_time,
            str(param.method),
            param.mass if param.mass is not None else np.ones(len(r), dtype=np.float64),
        )

    def trajectory(self, r: ArrF64, v: ArrF64, param: TrajectoryParam) -> tuple[ArrF64, ArrF64, float]:
        pass

import time

import numpy as np
from lite_dist2.table_node_api.start_table_api import start_in_thread

import rs_trijectory
from trijectory.engine.engine_param import TrajectoryParam
from trijectory.engine.python_engine import PythonEngine
from trijectory.node_ops import register_study, start_worker
from trijectory.plotter import plot_map, plot_trajectory


def run_life_gird_in_local() -> None:
    table_thread = start_in_thread()
    time.sleep(1)
    register_study("127.0.0.1")
    start_worker("127.0.0.1", stop_at_no_trial=True)

    table_thread.stop()
    table_thread.join(timeout=1)
    plot_map()


def run_trajectory_specific() -> None:
    sqrt3 = np.sqrt(3)
    r0 = np.array([[0, sqrt3 * 2 / 3], [-1, -sqrt3 / 3], [1, -sqrt3 / 3]], dtype=np.float64)
    v0 = np.array([[sqrt3 * 2 / 3, 0], [-3 / 4, sqrt3 / 4], [-3 / 4, -sqrt3 / 4]], dtype=np.float64) * 0.5
    ma = np.ones(len(r0), dtype=np.float64)

    _param = TrajectoryParam(
        max_time=3.0,
        time_step=0.0001,
        log_rate=100,
        escape_debounce_time=0.3,
        min_distance=0.01,
        method="rk44",
        mass=ma,
    )
    trajectory, bound_energy, life = PythonEngine().trajectory(r0, v0, _param)
    plot_trajectory(trajectory, bound_energy, _param)


def run_rust_code() -> None:
    r0 = np.array([[0, np.sqrt(3) * 2 / 3], [-1, -np.sqrt(3) / 3], [1, -np.sqrt(3) / 3]], dtype=np.float64)
    result = rs_trijectory.calc_relative_vector(r0)
    print(f"1 + 3 = {result}")  # noqa: T201


if __name__ == "__main__":
    run_rust_code()
    run_life_gird_in_local()

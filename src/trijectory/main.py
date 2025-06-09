import matplotlib.pyplot as plt
import numpy as np

from trijectory.engine.engine_param import TrajectoryParam
from trijectory.engine.python_engine import PythonEngine

if __name__ == "__main__":
    sqrt3 = np.sqrt(3)
    r0 = np.array([[0, sqrt3 * 2 / 3], [-1, -sqrt3 / 3], [1, -sqrt3 / 3]], dtype=np.float64)
    v0 = np.array([[sqrt3 * 2 / 3, 0], [-3 / 4, sqrt3 / 4], [-3 / 4, -sqrt3 / 4]], dtype=np.float64) * 0.75
    ma = np.ones(len(r0), dtype=np.float64)

    _param = TrajectoryParam(
        max_time=3.0,
        time_step=0.0001,
        log_rate=100,
        escape_debounce_time=0.3,
        method="rk44",
        mass=ma,
    )
    trajectory, bound_energy, life = PythonEngine().trajectory(r0, v0, _param)

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False)
    colors = ("r", "g", "b")
    for body_i in range(len(r0)):
        axes[0].plot(
            trajectory[:, 0, body_i, 0],
            trajectory[:, 0, body_i, 1],
            ls="-",
            marker=None,
            color=colors[body_i],
            label=str(body_i),
        )
    axes[0].legend()

    axes[1].plot(
        np.linspace(0, len(bound_energy) * _param.time_step * _param.log_rate, len(bound_energy)),
        bound_energy,
        color="k",
        label="rk44 be",
    )
    axes[1].legend()
    plt.show()

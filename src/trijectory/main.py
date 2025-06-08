import numpy as np
from tqdm import tqdm

from trijectory.engine.engine_param import TrajectoryParam
from trijectory.engine.python_engine import Euler, NumericalMethodsForODE, RungeKutta22, RungeKutta44
from trijectory.type_aliases import ArrF64


def calc_relative_vector(r: ArrF64) -> ArrF64:  # (body, body, space)
    # r = [[x0, y0],
    #      [x1, y1],
    #      [x2, y2]]
    return r[np.newaxis, :, :] - r[:, np.newaxis, :]


def calc_vectored_inv_square(r: ArrF64) -> ArrF64:  # (body, body, space) -> (body, body, space)
    rel = calc_relative_vector(r)
    non_zero_dist = np.linalg.norm(rel, axis=2) + np.eye(len(r))
    return rel / np.power(non_zero_dist[:, :, np.newaxis], 3)


def f(r: ArrF64, v: ArrF64, mass: ArrF64) -> tuple[ArrF64, ArrF64]:  # ((body, 2, space), (body,)) -> (body, 2, space)
    inv_square = calc_vectored_inv_square(r)
    ret_r = np.empty(r.shape)
    ret_v = np.empty(v.shape)
    for i_body in range(len(inv_square)):
        ret_r[i_body] = v[i_body]
        fv = np.sum(inv_square[i_body] * mass[:, np.newaxis], axis=0)
        ret_v[i_body] = fv
    return ret_r, ret_v


def run(
    solver: NumericalMethodsForODE,
    param: TrajectoryParam,
    r: ArrF64,
    v: ArrF64,
) -> ArrF64:
    iterations = int(param.max_time / param.time_step)
    log_arr = np.zeros((iterations + 1, 2, *r.shape))  # (step, 2, body, space)
    log_arr[0] = np.array([r, v])
    _r = r
    _v = v
    m = np.ones(len(r), dtype=np.float64) if param.mass is None else param.mass

    for step_i in tqdm(range(iterations)):
        _r, _v = solver.step(_r, _v, m, param.time_step)
        log_arr[step_i + 1, 0] = _r
        log_arr[step_i + 1, 1] = _v
    return log_arr


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r0 = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)  # (body, space)
    v0 = np.array([[0, 0], [0, 0], [0.3, 0]], dtype=np.float64)  # (body, space)
    ma = np.ones(len(r0), dtype=np.float64)

    euler = Euler(f)
    rk22 = RungeKutta22(f)
    rk44 = RungeKutta44(f)
    _param = TrajectoryParam(
        max_time=0.79,
        time_step=0.005,
        mass=ma,
    )
    log_arr_euler = run(euler, _param, r0, v0)
    log_arr_rk22 = run(rk22, _param, r0, v0)
    log_arr_rk44 = run(rk44, _param, r0, v0)

    colors = ("r", "g", "b")
    for body_i in range(len(r0)):
        plt.plot(
            log_arr_euler[:, 0, body_i, 0],
            log_arr_euler[:, 0, body_i, 1],
            ls="-",
            marker=None,
            color=colors[body_i],
            alpha=0.2,
            label="euler" + str(body_i),
        )
        plt.plot(
            log_arr_rk22[:, 0, body_i, 0],
            log_arr_rk22[:, 0, body_i, 1],
            ls="-",
            marker=None,
            color=colors[body_i],
            alpha=0.5,
            label="rk22" + str(body_i),
        )
        plt.plot(
            log_arr_rk44[:, 0, body_i, 0],
            log_arr_rk44[:, 0, body_i, 1],
            ls="-",
            marker=None,
            color=colors[body_i],
            alpha=1.0,
            label="rk44" + str(body_i),
        )
    plt.legend()
    plt.show()

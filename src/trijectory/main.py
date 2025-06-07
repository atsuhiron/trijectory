import numpy as np

from trijectory.engine.python_engine import Euler
from trijectory.type_aliases import ArrF64


def calc_relative_vector(r: ArrF64) -> ArrF64:  # (body, body, space)
    # r = [[x0, y0],
    #      [x1, y1],
    #      [x2, y2]]
    return r[:, np.newaxis, :] - r[np.newaxis, :, :]


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r0 = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)  # (body, space)
    v0 = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.float64)  # (body, space)
    ma = np.ones(len(r0), dtype=np.float64)

    euler = Euler(f)
    step = 40
    log_arr = np.zeros((step + 1, 2, *r0.shape))  # (step, 2, body, space)
    log_arr[0] = np.array([r0, v0])
    _r = r0
    _v = v0
    delta_t = 0.02
    for step_i in range(step):
        _r, _v = euler.step(_r, _v, ma, delta_t)
        log_arr[step_i + 1, 0] = _r
        log_arr[step_i + 1, 1] = _v

    for body_i in range(len(r0)):
        plt.plot(
            log_arr[:, 0, body_i, 0],
            log_arr[:, 0, body_i, 1],
            ls="-",
            marker="x",
            label=str(body_i),
        )
    plt.legend()
    plt.show()

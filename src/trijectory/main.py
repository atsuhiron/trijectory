import numpy as np

from trijectory.type_aliases import Arr64


def calc_relative_vector(r: Arr64) -> Arr64:  # (body, body, spa_axis)
    # r = [[x0, y0],
    #      [x1, y1],
    #      [x2, y2]]
    return r[:, np.newaxis, :] - r[np.newaxis, :, :]


def calc_vectored_inv_square(r: Arr64) -> Arr64:  # (body, body, spa_axis) -> (body, body, spa_axis)
    rel = calc_relative_vector(r)
    non_zero_dist = np.linalg.norm(rel, axis=2) + np.eye(len(r))
    return rel / np.power(non_zero_dist[:, :, np.newaxis], 3)


def f(rv: Arr64, mass: Arr64) -> Arr64:  # (body, 2, spa_axis), (body,) -> (body, 2, spa_axis)
    inv_square = calc_vectored_inv_square(rv[0])
    ret = np.empty(rv.shape)
    for i_body in range(len(rv)):
        ret[i_body, 0] = rv[i_body, 1]
        ret[i_body, 1] = np.sum(inv_square[i_body] * mass[:, np.newaxis], axis=1)
    return ret


if __name__ == "__main__":
    r0 = np.array([[0, 0], [10, 0]], dtype=np.float64)  # (body, spa_axis)
    v0 = np.array([[0, -1], [0, 1]], dtype=np.float64)
    rv0 = np.array([r0, v0])
    ma = np.ones(len(r0), dtype=np.float64)
    print(rv0[1])
    print(f(rv0, ma)[1])

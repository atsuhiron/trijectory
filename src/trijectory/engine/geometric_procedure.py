import numpy as np

from trijectory.type_aliases import ArrF64


def calc_relative_vector(r: ArrF64) -> ArrF64:  # (body, space) -> (body, body, space)
    # r = [[x0, y0],
    #      [x1, y1],
    #      [x2, y2]]
    return r[np.newaxis, :, :] - r[:, np.newaxis, :]


def calc_vectored_inv_square(r: ArrF64) -> ArrF64:  # (body, space) -> (body, body, space)
    rel = calc_relative_vector(r)
    non_zero_dist = np.linalg.norm(rel, axis=2) + np.eye(len(r))
    return rel / np.power(non_zero_dist[:, :, np.newaxis], 3)

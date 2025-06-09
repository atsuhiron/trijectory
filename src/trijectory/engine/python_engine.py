import abc
from collections.abc import Callable

import numpy as np

from trijectory.engine.base_engine import BaseEngine
from trijectory.engine.engine_param import TrajectoryParam
from trijectory.type_aliases import ArrF64


class NumericalMethodsForODE(metaclass=abc.ABCMeta):
    def __init__(self, func: Callable[[ArrF64, ArrF64, ArrF64], tuple[ArrF64, ArrF64]]) -> None:
        self.func = func

    @abc.abstractmethod
    def step(self, r: ArrF64, v: ArrF64, mass: ArrF64, delta_t: float) -> tuple[ArrF64, ArrF64]:
        pass


class Euler(NumericalMethodsForODE):
    def step(self, r: ArrF64, v: ArrF64, mass: ArrF64, delta_t: float) -> tuple[ArrF64, ArrF64]:
        delta_r, delta_v = self.func(r, v, mass)
        return r + delta_r * delta_t, v + delta_v * delta_t


class RungeKutta22(NumericalMethodsForODE):
    def step(self, r: ArrF64, v: ArrF64, mass: ArrF64, delta_t: float) -> tuple[ArrF64, ArrF64]:
        k1_r, k1_v = self.func(r, v, mass)
        k2_r, k2_v = self.func(r + k1_r * delta_t, v + k1_v * delta_t, mass)
        return r + delta_t * (k1_r + k2_r) / 2, v + delta_t * (k1_v + k2_v) / 2


class RungeKutta44(NumericalMethodsForODE):
    def step(self, r: ArrF64, v: ArrF64, mass: ArrF64, delta_t: float) -> tuple[ArrF64, ArrF64]:
        k1_r, k1_v = self.func(r, v, mass)
        k2_r, k2_v = self.func(r + 0.5 * k1_r * delta_t, v + 0.5 * k1_v * delta_t, mass)
        k3_r, k3_v = self.func(r + 0.5 * k2_r * delta_t, v + 0.5 * k2_v * delta_t, mass)
        k4_r, k4_v = self.func(r + k3_r * delta_t, v + k3_v * delta_t, mass)
        return r + delta_t * (k1_r / 6 + k2_r / 3 + k3_r / 3 + k4_r / 6), v + delta_t * (
            k1_v / 6 + k2_v / 3 + k3_v / 3 + k4_v / 6
        )


def calc_relative_vector(r: ArrF64) -> ArrF64:  # (body, space) -> (body, body, space)
    # r = [[x0, y0],
    #      [x1, y1],
    #      [x2, y2]]
    return r[np.newaxis, :, :] - r[:, np.newaxis, :]


def calc_vectored_inv_square(r: ArrF64) -> ArrF64:  # (body, space) -> (body, body, space)
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


def calc_bound_energy(r: ArrF64, v: ArrF64, mass: ArrF64) -> float:
    rel = calc_relative_vector(r)
    dist = np.linalg.norm(rel, axis=2)
    dist = np.array([dist[0, 1], dist[0, 2], dist[1, 2]])
    min_idx = np.argmin(dist)
    if min_idx == 0:
        e_3 = 2
        b_12 = (0, 1)
    elif min_idx == 1:
        e_3 = 1
        b_12 = (0, 2)
    else:
        e_3 = 0
        b_12 = (1, 2)

    # binary coordination, velocity
    binary_mass = mass[b_12[0]] + mass[b_12[1]]
    bound_r = (r[b_12[0]] * mass[b_12[0]] + r[b_12[1]] * mass[b_12[1]]) / binary_mass
    bound_v = (v[b_12[0]] * mass[b_12[0]] + v[b_12[1]] * mass[b_12[1]]) / binary_mass

    # relative of 3
    rel_r = r[e_3] - bound_r
    rel_v = v[e_3] - bound_v

    # Energy per unit mass
    energy_k = 0.5 * np.linalg.norm(rel_v) ** 2
    energy_u = binary_mass / np.linalg.norm(rel_r)
    return energy_k - energy_u


class DebounceEscapeDetector:
    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self._status = False
        self._step_count = 0

    def detect(self, status: bool) -> bool:
        if status:
            self._status = True
            self._step_count += 1
        else:
            self._status = False
            self._step_count = 0
        return self._step_count >= self.max_steps


class PythonEngine(BaseEngine):
    @staticmethod
    def create_solver(param: TrajectoryParam) -> NumericalMethodsForODE:
        if param.method == "euler":
            return Euler(f)
        if param.method == "rk22":
            return RungeKutta22(f)
        if param.method == "rk44":
            return RungeKutta44(f)

        msg = f"Unsupported method {param.method}"
        raise ValueError(msg)

    @staticmethod
    def run(
        solver: NumericalMethodsForODE,
        param: TrajectoryParam,
        r: ArrF64,
        v: ArrF64,
    ) -> tuple[ArrF64, ArrF64, int]:
        iterations = int(param.max_time / param.time_step)
        log_size = (iterations + 1) // param.log_rate

        m = np.ones(len(r), dtype=np.float64) if param.mass is None else param.mass
        trajectory = np.zeros((log_size, 2, *r.shape))  # (step, 2, body, space)
        bound_energy = np.zeros(log_size)  # (step,)
        ded = DebounceEscapeDetector(int(param.escape_debounce_time / param.time_step))

        trajectory[0] = np.array([r, v])
        bound_energy[0] = calc_bound_energy(r, v, m)
        _r = r
        _v = v

        for step_i in range(iterations):
            _r, _v = solver.step(_r, _v, m, param.time_step)
            _be = calc_bound_energy(_r, _v, m)
            if step_i % param.log_rate == 0:
                log_step = (step_i + 1) // param.log_rate
                trajectory[log_step, 0] = _r
                trajectory[log_step, 1] = _v
                bound_energy[log_step] = _be
            if ded.detect(_be > 0):
                return trajectory, bound_energy, step_i + 1

        return trajectory, bound_energy, iterations + 1

    def life(self, r: ArrF64, v: ArrF64, param: TrajectoryParam) -> float:
        solver = self.create_solver(param)
        _1, _2, iter_num = self.run(solver, param, r, v)
        return iter_num * param.time_step

    def trajectory(self, r: ArrF64, v: ArrF64, param: TrajectoryParam) -> tuple[ArrF64, ArrF64, float]:
        solver = self.create_solver(param)
        trajectory, bound_energy, iter_num = self.run(solver, param, r, v)
        log_size = iter_num // param.log_rate
        return trajectory[:log_size], bound_energy[:log_size], iter_num * param.time_step

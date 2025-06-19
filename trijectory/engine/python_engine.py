import abc
from collections.abc import Callable

import numpy as np

import trijectory.engine.geometric_procedure as geo_func
from trijectory.engine.base_engine import BaseEngine
from trijectory.engine.debounce_metric import CollisionMetric, EscapeMetric
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


def f(r: ArrF64, v: ArrF64, mass: ArrF64) -> tuple[ArrF64, ArrF64]:  # ((body, 2, space), (body,)) -> (body, 2, space)
    inv_square = geo_func.calc_vectored_inv_square(r)
    ret_r = np.empty(r.shape)
    ret_v = np.empty(v.shape)
    for i_body in range(len(inv_square)):
        ret_r[i_body] = v[i_body]
        fv = np.sum(inv_square[i_body] * mass[:, np.newaxis], axis=0)
        ret_v[i_body] = fv
    return ret_r, ret_v


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
        metric = np.zeros(log_size)  # (step,)
        escape_metrics = EscapeMetric(int(param.escape_debounce_time / param.time_step))
        collision_metrics = CollisionMetric(param.min_distance)
        metrics = (escape_metrics, collision_metrics)

        trajectory[0] = np.array([r, v])
        metric[0] = metrics[1].measure(r, v, m)
        _r = r
        _v = v

        for step_i in range(iterations):
            _r, _v = solver.step(_r, _v, m, param.time_step)
            if step_i % param.log_rate == 0:
                log_step = (step_i + 1) // param.log_rate
                trajectory[log_step, 0] = _r
                trajectory[log_step, 1] = _v
                metric[log_step] = metrics[1].measure(_r, _v, m)
            if any(met.detect(_r, _v, m) for met in metrics):
                return trajectory, metric, step_i + 1

        return trajectory, metric, iterations + 1

    @staticmethod
    def run_without_traj(
        solver: NumericalMethodsForODE,
        param: TrajectoryParam,
        r: ArrF64,
        v: ArrF64,
    ) -> int:
        iterations = int(param.max_time / param.time_step)

        m = np.ones(len(r), dtype=np.float64) if param.mass is None else param.mass
        escape_metrics = EscapeMetric(int(param.escape_debounce_time / param.time_step))
        collision_metrics = CollisionMetric(param.min_distance)
        metrics = (escape_metrics, collision_metrics)
        _r = r
        _v = v

        for step_i in range(iterations):
            _r, _v = solver.step(_r, _v, m, param.time_step)
            if any(met.detect(_r, _v, m) for met in metrics):
                return step_i + 1

        return iterations + 1

    def life(self, r: ArrF64, v: ArrF64, param: TrajectoryParam) -> float:
        solver = self.create_solver(param)
        iter_num = self.run_without_traj(solver, param, r, v)
        return iter_num * param.time_step

    def trajectory(self, r: ArrF64, v: ArrF64, param: TrajectoryParam) -> tuple[ArrF64, ArrF64, float]:
        solver = self.create_solver(param)
        trajectory, bound_energy, iter_num = self.run(solver, param, r, v)
        log_size = iter_num // param.log_rate
        return trajectory[:log_size], bound_energy[:log_size], iter_num * param.time_step

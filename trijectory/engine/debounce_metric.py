import abc
from typing import Literal

import numpy as np

import trijectory.engine.geometric_procedure as geo_func
from trijectory.type_aliases import ArrF64


class Metric(abc.ABC):
    def __init__(self, max_steps: int, threshold: float, ope: Literal["le", "lt", "ge", "gt"]) -> None:
        self.max_steps = max_steps
        self.threshold = threshold
        self.ope = ope
        self._status = False
        self._step_count = 0

    def detect(self, r: ArrF64, v: ArrF64, mass: ArrF64) -> bool:
        status = self.operate(self.measure(r, v, mass))
        if status:
            self._status = True
            self._step_count += 1
        else:
            self._status = False
            self._step_count = 0
        return self._step_count >= self.max_steps

    def operate(self, measure_value: float) -> bool:
        match self.ope:
            case "le":
                return measure_value <= self.threshold
            case "lt":
                return measure_value < self.threshold
            case "ge":
                return measure_value >= self.threshold
            case "gt":
                return measure_value > self.threshold
            case _:
                raise ValueError

    @abc.abstractmethod
    def measure(self, r: ArrF64, v: ArrF64, mass: ArrF64) -> float:
        pass


class EscapeMetric(Metric):
    def __init__(self, max_steps: int) -> None:
        super().__init__(max_steps, 0, "gt")

    def measure(self, r: ArrF64, v: ArrF64, mass: ArrF64) -> float:
        rel = geo_func.calc_relative_vector(r)
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


class CollisionMetric(Metric):
    def __init__(self, min_distance: float) -> None:
        super().__init__(1, min_distance, "lt")

    def measure(self, r: ArrF64, _v: ArrF64, _mass: ArrF64) -> float:
        rel = geo_func.calc_relative_vector(r)
        dist = np.linalg.norm(rel, axis=2)
        flatten_dist = dist[np.triu_indices(dist.shape[0], k=1)]
        return min(flatten_dist)

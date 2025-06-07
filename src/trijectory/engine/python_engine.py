import abc
from collections.abc import Callable

from trijectory.type_aliases import Arr64


class NumericalMethodsForODE(metaclass=abc.ABCMeta):
    def __init__(self, func: Callable[[Arr64, Arr64, Arr64], tuple[Arr64, Arr64]]) -> None:
        self.func = func

    @abc.abstractmethod
    def step(self, r: Arr64, v: Arr64, mass: Arr64, delta_t: float) -> tuple[Arr64, Arr64]:
        pass


class Euler(NumericalMethodsForODE):
    def step(self, r: Arr64, v: Arr64, mass: Arr64, delta_t: float) -> tuple[Arr64, Arr64]:
        delta_r, delta_v = self.func(r, v, mass)
        return r + delta_r * delta_t, v + delta_v * delta_t

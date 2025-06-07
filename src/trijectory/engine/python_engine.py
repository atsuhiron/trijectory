import abc
from collections.abc import Callable

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

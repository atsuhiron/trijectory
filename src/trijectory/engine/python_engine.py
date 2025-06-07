from collections.abc import Callable

from trijectory.type_aliases import Arr64


class Euler:
    def __init__(self, func: Callable[[Arr64, Arr64], Arr64]) -> None:
        self.func = func

    def step(self, rv: Arr64, mass: Arr64, delta_t: float) -> Arr64:
        return rv + self.func(rv, mass) * delta_t

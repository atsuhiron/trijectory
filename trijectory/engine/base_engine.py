import abc

from trijectory.engine.engine_param import TrajectoryParam
from trijectory.type_aliases import ArrF64


class BaseEngine(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def life(self, r: ArrF64, v: ArrF64, param: TrajectoryParam) -> float:
        pass

    @abc.abstractmethod
    def trajectory(self, r: ArrF64, v: ArrF64, param: TrajectoryParam) -> tuple[ArrF64, ArrF64, float]:
        pass

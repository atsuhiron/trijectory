import abc

from trijectory.engine.engine_param import TrajectoryParam
from trijectory.type_aliases import Arr64


class BaseEngine(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def life(self, x: Arr64, v: Arr64, param: TrajectoryParam) -> float:
        pass

    @abc.abstractmethod
    def trajectory(self, x: Arr64, v: Arr64, param: TrajectoryParam) -> Arr64:
        pass

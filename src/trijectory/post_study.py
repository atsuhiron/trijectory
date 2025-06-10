import argparse
import math

from lite_dist2.common import portablize
from lite_dist2.curriculum_models.study_portables import StudyRegistry
from lite_dist2.study_strategies import StudyStrategyModel
from lite_dist2.suggest_strategies import SuggestStrategyModel
from lite_dist2.suggest_strategies.base_suggest_strategy import SuggestStrategyParam
from lite_dist2.table_node_api.table_param import StudyRegisterParam
from lite_dist2.value_models.aligned_space_registry import LineSegmentRegistry, ParameterAlignedSpaceRegistry
from lite_dist2.worker_node.table_node_client import TableNodeClient


def _calc_start_and_step(center: float, half_width: float, size: int) -> tuple[float, float]:
    if size % 2 == 0:
        msg = "size must be odd"
        raise ValueError(msg)

    start = center - half_width
    step = 2 * half_width / (size - 1)
    return start, step


def register_study(table_ip: str) -> None:
    vx_size = 5
    vy_size = 5
    vx_start, vx_step = _calc_start_and_step(math.sqrt(3) * 2 / 3, 0.2, vx_size)
    vy_start, vy_step = _calc_start_and_step(0, 0.2, vy_size)

    study_register_param = StudyRegisterParam(
        study=StudyRegistry(
            name="trijectory",
            required_capacity=set(),
            study_strategy=StudyStrategyModel(
                type="all_calculation",
                study_strategy_param=None,
            ),
            suggest_strategy=SuggestStrategyModel(
                type="sequential",
                suggest_strategy_param=SuggestStrategyParam(strict_aligned=True),
            ),
            result_type="scalar",
            result_value_type="float",
            parameter_space=ParameterAlignedSpaceRegistry(
                type="aligned",
                axes=[
                    LineSegmentRegistry(
                        name="vx",
                        type="float",
                        size=str(portablize("int", vx_size)),
                        step=portablize("float", vx_step),
                        start=portablize("float", vx_start),
                    ),
                    LineSegmentRegistry(
                        name="vy",
                        type="float",
                        size=str(portablize("int", vy_size)),
                        step=portablize("float", vy_step),
                        start=portablize("float", vy_start),
                    ),
                ],
            ),
        ),
    )
    client = TableNodeClient(table_ip, name="admin node")
    client.register_study(study_register_param)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_ip")
    args = parser.parse_args()
    register_study(args.table_ip)

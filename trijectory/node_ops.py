import argparse
import json
import logging
import math
import shutil
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

import numpy as np
from lite_dist2.common import portablize
from lite_dist2.config import WorkerConfig
from lite_dist2.curriculum_models.study_portables import StudyRegistry
from lite_dist2.study_strategies import StudyStrategyModel
from lite_dist2.suggest_strategies import SuggestStrategyModel
from lite_dist2.suggest_strategies.base_suggest_strategy import SuggestStrategyParam
from lite_dist2.table_node_api.table_param import StudyRegisterParam
from lite_dist2.table_node_api.table_response import StudyRegisteredResponse
from lite_dist2.type_definitions import RawParamType, RawResultType
from lite_dist2.value_models.aligned_space_registry import LineSegmentRegistry, ParameterAlignedSpaceRegistry
from lite_dist2.worker_node.table_node_client import TableNodeClient
from lite_dist2.worker_node.trial_runner import SemiAutoMPTrialRunner
from lite_dist2.worker_node.worker import Worker

from trijectory.engine.engine_param import TrajectoryParam
from trijectory.engine.python_engine import PythonEngine
from trijectory.engine.rust_engine import RustEngine

logger = logging.getLogger(__name__)


class TrijectoryRunner(SemiAutoMPTrialRunner):
    def func(self, parameters: RawParamType, *_args: tuple[Any, ...], **_kwargs: dict[str, Any]) -> RawResultType:
        sqrt3 = np.sqrt(3)
        r0 = np.array([[0, sqrt3 * 2 / 3], [-1, -sqrt3 / 3], [1, -sqrt3 / 3]], dtype=np.float64)
        v0 = np.array([[*parameters], [-3 / 4, sqrt3 / 4], [-3 / 4, -sqrt3 / 4]], dtype=np.float64) * 0.75

        ma = np.ones(3, dtype=np.float64)
        _param = TrajectoryParam(
            max_time=3.0,
            time_step=0.0001,
            log_rate=100,
            escape_debounce_time=0.3,
            min_distance=0.01,
            method="rk44",
            mass=ma,
        )
        if _param.backend == "rust":
            return RustEngine().life(r0, v0, _param)
        if _param.backend == "python":
            return PythonEngine().life(r0, v0, _param)
        raise ValueError(f"Unknown backend {_param.backend}")


def _load_worker_config() -> WorkerConfig:
    config = Path(__file__).parent.parent / "worker_config.json"
    if not config.exists():
        default_config = Path(__file__).parent.parent / "default_worker_config.json"
        shutil.copyfile(default_config, config)
        new_name = input("Enter new worker config name: ")

        with config.open("r") as f:
            worker_config = json.load(f)
        worker_config["name"] = new_name
        with config.open("w") as f:
            json.dump(worker_config, f, indent=2)
    else:
        with config.open("r") as f:
            worker_config = json.load(f)
    return WorkerConfig.model_validate(worker_config)


def _calc_start_and_step(center: float, half_width: float, size: int) -> tuple[float, float]:
    if size % 2 == 0:
        msg = "size must be odd"
        raise ValueError(msg)

    start = center - half_width
    step = 2 * half_width / (size - 1)
    return start, step


def _register_study(table_ip: str) -> StudyRegisteredResponse:
    vx_size = 13
    vy_size = 13
    vx_start, vx_step = _calc_start_and_step(math.sqrt(3) * 2 / 3, 0.25, vx_size)
    vy_start, vy_step = _calc_start_and_step(0, 0.25, vy_size)

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
    client = TableNodeClient(table_ip, port=8000)
    return client.register_study(study_register_param)


def start_worker(table_ip: str | None = None, stop_at_no_trial: bool = False) -> None:
    if table_ip is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--table-ip", default=None, type=str)
        args = parser.parse_args()
        table_ip = args.table_ip
        if table_ip is None:
            raise ValueError("Set table node IP")

    config = _load_worker_config()

    with Pool(processes=config.process_num) as pool:
        worker = Worker(
            trial_runner=TrijectoryRunner(),
            ip=table_ip,
            port=8000,
            config=config,
            pool=pool,
        )
        worker.start(stop_at_no_trial=stop_at_no_trial)


def register_study(table_ip: str | None = None) -> None:
    if table_ip is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--table-ip", default=None, type=str)
        args = parser.parse_args()
        table_ip = args.table_ip
        if table_ip is None:
            raise ValueError("Set table node IP")

    registered_result = _register_study(table_ip)
    logger.info(registered_result.study_id)

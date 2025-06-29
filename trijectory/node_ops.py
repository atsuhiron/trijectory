import argparse
import json
import logging
import shutil
from multiprocessing.pool import Pool
from pathlib import Path

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
from lite_dist2.value_models.const_param import ConstParam
from lite_dist2.worker_node.table_node_client import TableNodeClient
from lite_dist2.worker_node.trial_runner import SemiAutoMPTrialRunner
from lite_dist2.worker_node.worker import Worker

from trijectory.engine.engine_param import TrajectoryParam
from trijectory.engine.python_engine import PythonEngine
from trijectory.engine.rust_engine import RustEngine

logger = logging.getLogger(__name__)


class TrijectoryRunner(SemiAutoMPTrialRunner):
    def func(self, parameters: RawParamType, *_args: object, **kwargs: object) -> RawResultType:
        sqrt3 = np.sqrt(3)
        y, vx, vy = parameters
        r0 = np.array([[0, y], [-1, -sqrt3 / 3], [1, -sqrt3 / 3]], dtype=np.float64)
        v0 = np.array([[vx, vy], [-3 / 4, sqrt3 / 4], [-3 / 4, -sqrt3 / 4]], dtype=np.float64) * 0.75

        ma = np.ones(3, dtype=np.float64)
        _param = TrajectoryParam(
            max_time=self.get_typed("max_time", float, kwargs),
            time_step=self.get_typed("time_step", float, kwargs),
            log_rate=self.get_typed("log_rate", int, kwargs),
            escape_debounce_time=self.get_typed("escape_debounce_time", float, kwargs),
            min_distance=self.get_typed("min_distance", float, kwargs),
            method="rk44",
            mass=ma,
        )
        if _param.backend == "rust":
            return RustEngine().life(r0, v0, _param)
        if _param.backend == "python":
            return PythonEngine().life(r0, v0, _param)
        msg = f"Unknown backend {_param.backend}"
        raise ValueError(msg)


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
    start = center - half_width
    step = 2 * half_width / (size - 1)
    return start, step


def _register_study(table_ip: str) -> StudyRegisteredResponse:
    vx_size = 30
    vy_size = 30
    y_size = 5
    vx_start, vx_step = _calc_start_and_step(0.7, 0.8, vx_size)
    vy_start, vy_step = _calc_start_and_step(0.35, 0.8, vy_size)
    y_start, y_step = _calc_start_and_step(np.sqrt(3) * 2 / 3, 0.1, y_size)
    const_param = {
        "max_time": 32.0,
        "time_step": 0.0001,
        "log_rate": 100,
        "escape_debounce_time": 0.3,
        "min_distance": 0.01,
    }

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
            const_param=ConstParam.from_dict(const_param),
            parameter_space=ParameterAlignedSpaceRegistry(
                type="aligned",
                axes=[
                    LineSegmentRegistry(
                        name="y",
                        type="float",
                        size=str(portablize("int", y_size)),
                        step=portablize("float", y_step),
                        start=portablize("float", y_start),
                    ),
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


def register_study(table_ip: str | None = None) -> str:
    if table_ip is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--table-ip", default=None, type=str)
        args = parser.parse_args()
        table_ip = args.table_ip
        if table_ip is None:
            raise ValueError("Set table node IP")

    registered_result = _register_study(table_ip)
    logger.info(registered_result.study_id)
    return registered_result.study_id

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from lite_dist2.config import WorkerConfig
from lite_dist2.type_definitions import RawParamType, RawResultType
from lite_dist2.worker_node.trial_runner import AutoMPTrialRunner
from lite_dist2.worker_node.worker import Worker

from trijectory.engine.engine_param import TrajectoryParam
from trijectory.engine.python_engine import PythonEngine


class TrijectoryRunner(AutoMPTrialRunner):
    def func(self, parameters: RawParamType, *_args: tuple[Any, ...], **_kwargs: dict[str, Any]) -> RawResultType:
        sqrt3 = np.sqrt(3)
        r0 = np.array([[0, sqrt3 * 2 / 3], [-1, -sqrt3 / 3], [1, -sqrt3 / 3]], dtype=np.float64)
        v0 = np.array([[*parameters], [-3 / 4, sqrt3 / 4], [-3 / 4, -sqrt3 / 4]], dtype=np.float64) * 0.75

        ma = np.ones(3, dtype=np.float64)
        _param = TrajectoryParam(
            max_time=2.0,
            time_step=0.0001,
            log_rate=100,
            escape_debounce_time=0.3,
            min_distance=0.01,
            method="rk44",
            mass=ma,
        )
        return PythonEngine().life(r0, v0, _param)


def _load_worker_config() -> WorkerConfig:
    config = Path(__file__).parent.parent.parent / "worker_config.json"
    if not config.exists():
        default_config = Path(__file__).parent.parent.parent / "default_worker_config.json"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_ip")
    args = parser.parse_args()

    config = _load_worker_config()
    worker = Worker(
        trial_runner=TrijectoryRunner(),
        ip=args.table_ip,
        port=8000,
        config=config,
    )
    worker.start()


if __name__ == "__main__":
    main()

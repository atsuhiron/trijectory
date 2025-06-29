import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lite_dist2.common import numerize
from lite_dist2.curriculum_models.curriculum import CurriculumModel
from lite_dist2.curriculum_models.study_portables import StudyStorage

from trijectory.engine.engine_param import TrajectoryParam
from trijectory.type_aliases import ArrF64


def _load_curriculum() -> CurriculumModel:
    with Path("curriculum.json").open("r") as f:
        json_data = json.load(f)
        return CurriculumModel.model_validate(json_data)


def _extract_storage(curriculum: CurriculumModel, study_id: str | None = None) -> StudyStorage:
    if len(curriculum.storages) == 0:
        msg = "No storage is found in curriculum"
        raise ValueError(msg)

    if study_id is None:
        return max(curriculum.storages, key=lambda s: s.registered_timestamp)
    storages = list(filter(lambda s: s.study_id == study_id, curriculum.storages))
    if len(storages) != 1:
        msg = "0 or more than one storage is found in curriculum"
        raise ValueError(msg)
    return storages[0]


def _load_storage(path: Path) -> StudyStorage:
    with path.open("r") as f:
        d = json.load(f)
        return StudyStorage.model_validate(d)


def _extract_result_arr(storage: StudyStorage) -> tuple[ArrF64, ArrF64, ArrF64]:
    xyz_list: list[tuple[float, float, float]] = [
        (
            numerize("float", xyz[0]),
            numerize("float", xyz[1]),
            numerize("float", xyz[-1]),
        )
        for xyz in storage.results.values
    ]
    xyz_list = sorted(xyz_list, key=lambda r: r[1])

    y_groupby_x_sorted_dict = {
        y: sorted(xz, key=lambda _xz: _xz[0]) for y, xz in itertools.groupby(xyz_list, lambda xyz: xyz[1])
    }
    sorted_y_key = sorted(y_groupby_x_sorted_dict.keys())
    shape = (len(y_groupby_x_sorted_dict), len(y_groupby_x_sorted_dict[sorted_y_key[0]]))
    arr = np.zeros(shape, dtype=np.float64)
    x_axis = np.array([tup[0] for tup in y_groupby_x_sorted_dict[sorted_y_key[0]]])
    y_axis = np.array(sorted_y_key)
    for i, y in enumerate(sorted_y_key):
        for j, xz in enumerate(y_groupby_x_sorted_dict[y]):
            arr[i, j] = xz[2]
    return arr, y_axis, x_axis


def _extract_axis_name(storage: StudyStorage) -> tuple[str | None, str | None]:
    param_info = storage.results.params_info
    return param_info[0].name, param_info[1].name


def _plot_map(arr: ArrF64, y_axis: ArrF64, x_axis: ArrF64, y_axis_name: str | None, x_axis_name: str | None) -> None:
    plt.imshow(
        arr,
        origin="lower",
        extent=(float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])),
    )
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.colorbar()
    plt.show()


def plot_map(file_path: Path | None = None, study_id: str | None = None) -> None:
    if study_id is not None:
        storage = _extract_storage(_load_curriculum(), study_id)
    elif file_path is not None and file_path.exists():
        storage = _load_storage(file_path)
    else:
        raise ValueError
    arr, y_axis, x_axis = _extract_result_arr(storage)
    x_name, y_name = _extract_axis_name(storage)
    _plot_map(arr, y_axis, x_axis, y_name, x_name)


def plot_trajectory(trajectory: ArrF64, metric: ArrF64, param: TrajectoryParam | None) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False)
    colors = ("r", "g", "b")
    for body_i in range(trajectory.shape[2]):  # (step, 2, body, space)
        axes[0].plot(
            trajectory[:, 0, body_i, 0],
            trajectory[:, 0, body_i, 1],
            ls="-",
            marker=None,
            color=colors[body_i],
            label=str(body_i),
        )
        axes[0].plot(
            [trajectory[0, 0, body_i, 0]],
            [trajectory[0, 0, body_i, 1]],
            marker="o",
            color=colors[body_i],
        )
    axes[0].legend()

    if param is None:
        x_range = np.arange(len(metric))
    else:
        x_range = np.linspace(0, len(metric) * param.time_step * param.log_rate, len(metric))
    axes[1].plot(
        x_range,
        metric,
        color="k",
        label="metric",
    )
    axes[1].legend()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None)
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    plot_map(file_path=Path(args.path), study_id=args.study_id)

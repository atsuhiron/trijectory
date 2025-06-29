import argparse
import itertools
import json
from pathlib import Path

import imageio
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
        return StudyStorage.model_validate(d["result"])


def _extract_result_core(
    y_groupby_x_sorted_dict: dict[float, list[tuple[float, float, float]]],
    shape: tuple[int, int],
) -> tuple[ArrF64, ArrF64, ArrF64]:
    sorted_y_key = sorted(y_groupby_x_sorted_dict.keys())
    arr = np.zeros(shape, dtype=np.float64)
    x_axis = np.array([tup[0] for tup in y_groupby_x_sorted_dict[sorted_y_key[0]]])
    y_axis = np.array(sorted_y_key)
    for i, y in enumerate(sorted_y_key):
        for j, xz in enumerate(y_groupby_x_sorted_dict[y]):
            arr[i, j] = xz[2]
    return arr, y_axis, x_axis


def _extract_result_arr2(values: list[tuple[bool | str, ...]]) -> tuple[ArrF64, ArrF64, ArrF64]:
    xyz_list: list[tuple[float, float, float]] = [
        (
            numerize("float", xyz[0]),
            numerize("float", xyz[1]),
            numerize("float", xyz[-1]),
        )
        for xyz in values
    ]
    xyz_list = sorted(xyz_list, key=lambda xyz: xyz[1])
    shape = (
        len({xyz[0] for xyz in xyz_list}),
        len({xyz[1] for xyz in xyz_list}),
    )
    y_groupby_x_sorted_dict = {
        y: sorted(xz, key=lambda _xz: _xz[0]) for y, xz in itertools.groupby(xyz_list, lambda xyz: xyz[1])
    }
    return _extract_result_core(y_groupby_x_sorted_dict, shape)


def _extract_result_arr3(values: list[tuple[bool | str, ...]]) -> tuple[ArrF64, ArrF64, list[ArrF64], list[ArrF64]]:
    wxyz_list: list[tuple[float, float, float, float]] = [
        (
            numerize("float", xyz[0]),
            numerize("float", xyz[1]),
            numerize("float", xyz[2]),
            numerize("float", xyz[-1]),
        )
        for xyz in values
    ]
    wxyz_list = sorted(wxyz_list, key=lambda wxyz: wxyz[0])
    shape = (
        len({wxyz[0] for wxyz in wxyz_list}),
        len({wxyz[1] for wxyz in wxyz_list}),
        len({wxyz[2] for wxyz in wxyz_list}),
    )
    shape_2d = (shape[1], shape[2])
    arr = np.zeros(shape, dtype=np.float64)
    w_groupby_dict = {w: list(xyz) for w, xyz in itertools.groupby(wxyz_list, lambda wxyz: wxyz[0])}
    w_axis = np.array(list(w_groupby_dict.keys()))
    y_axis_list = []
    x_axis_list = []
    for i, w in enumerate(w_axis):
        xyz_list = sorted([(wxyz[1], wxyz[2], wxyz[3]) for wxyz in w_groupby_dict[w]], key=lambda xyz: xyz[1])
        y_groupby_x_sorted_dict = {
            y: sorted(xz, key=lambda _xz: _xz[0]) for y, xz in itertools.groupby(xyz_list, lambda xyz: xyz[1])
        }
        arr_xy, y_axis, x_axis = _extract_result_core(y_groupby_x_sorted_dict, shape_2d)

        arr[i] = arr_xy
        y_axis_list.append(y_axis)
        x_axis_list.append(x_axis)

    return arr, w_axis, y_axis_list, x_axis_list


def _extract_axis_name(storage: StudyStorage) -> tuple[str | None, ...]:
    names = storage.results.get_names()
    return names[:-1]


def _plot_map_2d(
    arr: ArrF64,
    y_axis: ArrF64,
    x_axis: ArrF64,
    y_axis_name: str | None,
    x_axis_name: str | None,
    v_max: float | None = None,
    v_min: float | None = None,
    title: str | None = None,
    save_path: Path | None = None,
) -> None:
    if v_max is None:
        v_max = np.max(arr)
    if v_min is None:
        v_min = np.min(arr)

    plt.imshow(
        arr,
        origin="lower",
        extent=(float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])),
        vmax=v_max,
        vmin=v_min,
    )
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.colorbar()
    if title:
        plt.title(title)
    if save_path is None:
        plt.show()
        return
    with save_path.open("wb") as f:
        plt.savefig(f)


def _plot_map_3d(
    arr: ArrF64,
    w_axis: ArrF64,
    y_axis_list: list[ArrF64],
    x_axis_list: list[ArrF64],
    y_axis_name: str | None,
    x_axis_name: str | None,
    fps: float = 10.0,
) -> None:
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    v_max = np.max(arr)
    v_min = np.min(arr)
    image_paths: list[Path] = []
    frame_num = len(w_axis)
    digits = len(str(frame_num))
    for i, w in enumerate(w_axis):
        plt.clf()
        title_str = f"({i:{digits}}/{frame_num}) w={w:7.4f}"
        path = Path(f"temp/fig_{str(i).zfill(5)}.png")
        _plot_map_2d(arr[i], y_axis_list[i], x_axis_list[i], y_axis_name, x_axis_name, v_max, v_min, title_str, path)
        image_paths.append(path)

    with imageio.get_writer(Path("trijectory.gif"), mode="I", fps=fps, loop=4) as writer:
        for image_path in image_paths:
            image = imageio.v3.imread(image_path)
            writer.append_data(image)

    for image_path in image_paths:
        image_path.unlink()
    temp_dir.rmdir()


def plot_map(file_path: Path | None = None, study_id: str | None = None, fps: int = 10) -> None:
    if study_id is not None:
        storage = _extract_storage(_load_curriculum(), study_id)
    elif file_path is not None and file_path.exists():
        storage = _load_storage(file_path)
    else:
        raise ValueError

    match len(storage.results.get_types()):
        case 3:
            arr, y_axis, x_axis = _extract_result_arr2(storage.results.values)
            x_name, y_name = _extract_axis_name(storage)
            _plot_map_2d(arr, y_axis, x_axis, y_name, x_name)
        case 4:
            arr, w_axis, y_axis_list, x_axis_list = _extract_result_arr3(storage.results.values)
            w_name, x_name, y_name = _extract_axis_name(storage)
            _plot_map_3d(arr, w_axis, y_axis_list, x_axis_list, y_name, x_name, fps)
        case _:
            raise ValueError


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    plot_map(
        file_path=Path(args.path) if args.path is not None else None,
        study_id=args.id,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

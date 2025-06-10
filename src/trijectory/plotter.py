import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lite_dist2.common import numerize
from lite_dist2.curriculum_models.curriculum import CurriculumModel
from lite_dist2.curriculum_models.study_portables import StudyStorage
from lite_dist2.value_models.point import ScalarValue

from trijectory.type_aliases import ArrF64


def load_curriculum() -> CurriculumModel:
    with Path("curriculum.json").open("r") as f:
        json_data = json.load(f)
        return CurriculumModel.model_validate(json_data)


def extract_storage(curriculum: CurriculumModel, study_id: str | None = None) -> StudyStorage:
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


def extract_result_arr(storage: StudyStorage) -> tuple[ArrF64, ArrF64, ArrF64]:
    z_list = list(filter(lambda z: isinstance(z, ScalarValue), [r.result for r in storage.result]))
    xyz_list: list[tuple[float, float, float]] = [
        (
            numerize("float", r.params[0].value),
            numerize("float", r.params[1].value),
            numerize("float", z.value),
        )
        for r, z in zip(storage.result, z_list, strict=True)
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


def extract_axis_name(storage: StudyStorage) -> tuple[str | None, str | None]:
    mapping = storage.result[0]
    return mapping.params[0].name, mapping.params[1].name


def plot(arr: ArrF64, y_axis: ArrF64, x_axis: ArrF64, y_axis_name: str | None, x_axis_name: str | None) -> None:
    plt.imshow(arr)
    plt.xticks(ticks=np.arange(len(x_axis)), labels=np.round(x_axis, 2))
    plt.yticks(ticks=np.arange(len(y_axis)), labels=np.round(y_axis, 2))
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    _storage = extract_storage(load_curriculum())
    _arr, _y_axis, _x_axis = extract_result_arr(_storage)
    _y_name, _x_name = extract_axis_name(_storage)
    plot(_arr, _y_axis, _x_axis, _y_name, _x_name)

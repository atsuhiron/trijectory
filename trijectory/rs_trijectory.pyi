from trijectory.type_aliases import ArrF64


def add(a: int, b: int) -> int: ...


def calc_life(
    r: ArrF64,
    v: ArrF64,
    max_time: float,
    time_step: float,
    escape_debounce_time: float,
    min_distance: float,
    method_str: str,
    m: ArrF64,
) -> float: ...

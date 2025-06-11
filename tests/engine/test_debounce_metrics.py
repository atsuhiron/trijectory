import numpy as np

from trijectory.engine.debounce_metric import CollisionMetric, EscapeMetric
from trijectory.type_aliases import ArrF64


def _create_data() -> tuple[ArrF64, ArrF64, ArrF64]:
    return (
        np.array(
            [
                [0.0, 0.0, 0.0],  # Body 1 at origin
                [1.0, 0.0, 0.0],  # Body 2 at (1,0,0)
                [10.0, 10.0, 10.0],  # Body 3 far away
            ],
        ),
        np.array(
            [
                [0.0, 0.0, 0.0],  # Body 1 stationary
                [0.0, 1.0, 0.0],  # Body 2 moving in y direction
                [5.0, 5.0, 5.0],  # Body 3 moving away fast
            ],
        ),
        np.array([1.0, 1.0, 0.5]),  # Masses of the three bodies
    )


def test_escape_metrics_measure() -> None:
    # Create an instance of EscapeMetric
    escape_metric = EscapeMetric(max_steps=3)

    # Create test data
    # Three bodies with positions, velocities, and masses
    # Body 1 and 2 form a binary system, body 3 is escaping
    r, v, mass = _create_data()

    # Calculate the measure
    result = escape_metric.measure(r, v, mass)

    # For an escaping body, the total energy (kinetic - potential) should be positive
    assert result > 0, "Expected positive energy for escaping body"

    # Now test with a non-escaping scenario
    # Body 3 is closer and moving slower
    r_bound = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
    )

    v_bound = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
    )

    # Calculate the measure for the bound case
    result_bound = escape_metric.measure(r_bound, v_bound, mass)

    # For a bound body, the total energy should be negative
    assert result_bound < 0, "Expected negative energy for bound body"


def test_escape_metrics_detect() -> None:
    # Create an instance of EscapeMetric with max_steps=2
    escape_metric = EscapeMetric(max_steps=2)

    # Create test data for an escaping body
    r, v, mass = _create_data()

    # First detection should return False as it hasn't reached max_steps
    assert not escape_metric.detect(r, v, mass), "First detection should return False"

    # Second detection should return True as it has reached max_steps
    assert escape_metric.detect(r, v, mass), "Second detection should return True"

    # Now test with a non-escaping scenario
    r_bound = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
    )

    v_bound = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
    )

    # Reset the metric
    escape_metric = EscapeMetric(max_steps=2)

    # Detection should return False for bound body
    assert not escape_metric.detect(r_bound, v_bound, mass), "Detection should return False for bound body"

    # Even second detection should return False
    assert not escape_metric.detect(r_bound, v_bound, mass), "Second detection should also return False for bound body"


def _create_data_no_collision() -> tuple[ArrF64, ArrF64, ArrF64]:
    return (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        ),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ),
        np.array([1.0, 1.0, 1.0]),
    )


def test_collision_metrics_measure() -> None:
    # Create an instance of CollisionMetric
    collision_metric = CollisionMetric(min_distance=0.5)

    # Create test data
    # Three bodies with positions
    r_no_collision, v, mass = _create_data_no_collision()

    # Calculate the minimum distance
    min_dist = collision_metric.measure(r_no_collision, v, mass)

    # The minimum distance should be 1.0
    assert min_dist == 1.0, f"Expected minimum distance of 1.0, got {min_dist}"

    # Now test with a potential collision
    r_collision = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],  # Distance to body 1 is 0.4
            [2.0, 0.0, 0.0],
        ],
    )

    # Calculate the minimum distance for the collision case
    min_dist_collision = collision_metric.measure(r_collision, v, mass)

    # The minimum distance should be 0.4
    assert min_dist_collision == 0.4, f"Expected minimum distance of 0.4, got {min_dist_collision}"


def test_collision_metrics_detect() -> None:
    # Create an instance of CollisionMetric with min_distance=0.5
    collision_metric = CollisionMetric(min_distance=0.5)

    # Create test data for no collision
    r_no_collision, v, mass = _create_data_no_collision()

    # Detection should return False as minimum distance (1.0) > threshold (0.5)
    assert not collision_metric.detect(r_no_collision, v, mass), "Detection should return False for no collision"

    # Now test with a collision scenario
    r_collision = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],  # Distance to body 1 is 0.4
            [2.0, 0.0, 0.0],
        ],
    )

    # Detection should return True as minimum distance (0.4) < threshold (0.5)
    assert collision_metric.detect(r_collision, v, mass), "Detection should return True for collision"

    # Test with a borderline case
    r_borderline = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # Distance to body 1 is exactly 0.5
            [2.0, 0.0, 0.0],
        ],
    )

    # Detection should return True as minimum distance (0.5) == threshold (0.5) and operator is "lt"
    assert not collision_metric.detect(r_borderline, v, mass), "Detection should return False for borderline case"

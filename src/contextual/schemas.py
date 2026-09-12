from dataclasses import dataclass, field
from typing import Dict, List


CORE_VARIABLES = ["temperature", "pressure", "humidity"]

REQUIRED_CURRENT_FIELDS = [
    "station_id",
    "timestamp",
    "temperature",
    "pressure",
    "humidity",
]

REQUIRED_ANOMALY_FIELDS = [
    "station_id",
    "timestamp",
    "anomaly_score",
    "anomaly_status",
]


@dataclass
class ContextConfig:
    """
    Configuration for contextual analysis.

    Thresholds are intentionally configurable because the correct values
    depend on the AWS sampling interval and real station data.
    """

    # Rate thresholds are expressed per minute.
    rapid_rate_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "temperature": 1.0,   # °C/min
            "pressure": 0.5,      # hPa/min
            "humidity": 3.0,      # percentage points/min
        }
    )

    # Absolute changes used to determine whether a variable changed
    # substantially between observations.
    significant_change_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "temperature": 1.5,
            "pressure": 1.0,
            "humidity": 5.0,
        }
    )

    # Number of recent observations used for persistence/stuck analysis.
    persistence_window: int = 3

    # Number of observations needed for drift analysis.
    drift_window: int = 5

    # A sensor is considered "stuck" only when values vary less than
    # these tolerances over the stuck window.
    stuck_tolerance: Dict[str, float] = field(
        default_factory=lambda: {
            "temperature": 0.01,
            "pressure": 0.01,
            "humidity": 0.01,
        }
    )

    # Minimum slope for a basic drift indicator.
    drift_rate_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "temperature": 0.02,
            "pressure": 0.02,
            "humidity": 0.05,
        }
    )

    # Minimum standard deviation before calculating a z-score.
    minimum_std: float = 1e-9


def validate_current_observation(data: dict) -> List[str]:
    """Return missing required fields in a Layer 3 observation."""

    return [
        field
        for field in REQUIRED_CURRENT_FIELDS
        if field not in data
    ]


def validate_anomaly_result(data: dict) -> List[str]:
    """Return missing required fields in a Layer 4 anomaly result."""

    return [
        field
        for field in REQUIRED_ANOMALY_FIELDS
        if field not in data
    ]
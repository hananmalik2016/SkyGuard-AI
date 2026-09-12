import numpy as np
import pandas as pd

from .schemas import CORE_VARIABLES, ContextConfig


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_rate(current_data, variable):
    """
    Prefer the rate already calculated by Layer 3.
    """
    rate = _safe_float(
        current_data.get(f"{variable}_rate")
    )

    if rate is not None:
        return rate

    # Fallback: calculate from diff_1 and time_diff.
    diff = _safe_float(
        current_data.get(f"{variable}_diff_1")
    )

    time_diff = _safe_float(
        current_data.get("time_diff")
    )

    if diff is None or time_diff is None or time_diff <= 0:
        return None

    # Assume time_diff is expressed in seconds.
    return diff / (time_diff / 60.0)


def _get_recent_values(
    recent_data: pd.DataFrame,
    variable: str,
    window: int,
):
    if recent_data is None or recent_data.empty:
        return []

    if variable not in recent_data.columns:
        return []

    values = (
        pd.to_numeric(
            recent_data[variable],
            errors="coerce"
        )
        .dropna()
        .tolist()
    )

    return values[-window:]


def _persistence_indicator(
    values,
    config: ContextConfig,
    variable: str,
):
    """
    Determine whether a change appears persistent.

    We compare the latest value against the earlier values.
    """

    if len(values) < config.persistence_window:
        return None

    values = np.asarray(values, dtype=float)

    baseline = np.median(
        values[:-config.persistence_window + 1]
    )

    recent = values[-config.persistence_window:]

    threshold = config.significant_change_thresholds[variable]

    deviations = recent - baseline

    # All recent values are substantially above baseline.
    if np.all(deviations > threshold):
        return True

    # All recent values are substantially below baseline.
    if np.all(deviations < -threshold):
        return True

    return False


def _stuck_indicator(values, tolerance):
    """
    Detect nearly identical consecutive values.

    This is only an indicator. A stable real environment can also
    produce constant values.
    """

    if len(values) < 3:
        return None

    values = np.asarray(values, dtype=float)

    return bool(
        np.max(values) - np.min(values) <= tolerance
    )


def _drift_indicator(values, threshold):
    if len(values) < 4:
        return None

    values = np.asarray(values, dtype=float)

    differences = np.diff(values)

    if len(differences) == 0:
        return {
            "detected": False,
            "slope_per_observation": 0.0,
        }

    positive_ratio = np.mean(differences > 0)
    negative_ratio = np.mean(differences < 0)

    same_direction = bool(
    max(
        positive_ratio,
        negative_ratio
    ) >= 0.75
    )

    x = np.arange(len(values))
    slope = float(np.polyfit(x, values, 1)[0])

    # Compare the largest single change with the
    # typical change in the window.
    typical_change = float(
        np.median(np.abs(differences))
    )

    largest_change = float(
        np.max(np.abs(differences))
    )

    # A very large single jump should not be interpreted
    # as gradual drift.
    jump_dominates = (
        typical_change > 0
        and largest_change > 3 * typical_change
    )

    detected = (
        same_direction
        and abs(slope) >= threshold
        and not jump_dominates
    )

    return {
        "detected": detected,
        "slope_per_observation": slope,
        "same_direction": same_direction,
        "jump_dominates": jump_dominates,
    }

def analyze_rate_and_persistence(
    current_data: dict,
    recent_data: pd.DataFrame,
    config: ContextConfig | None = None,
) -> dict:

    config = config or ContextConfig()

    result = {}

    for variable in CORE_VARIABLES:

        rate = _get_rate(
            current_data,
            variable
        )

        rapid_change = (
            rate is not None
            and abs(rate)
            >= config.rapid_rate_thresholds[variable]
        )

        values = _get_recent_values(
            recent_data,
            variable,
            max(
                config.persistence_window,
                config.drift_window
            ),
        )

        persistence = _persistence_indicator(
            values,
            config,
            variable,
        )

        stuck = _stuck_indicator(
            values[-config.persistence_window:],
            config.stuck_tolerance[variable],
        )

        drift = _drift_indicator(
            values[-config.drift_window:],
            config.drift_rate_thresholds[variable],
        )

        result[variable] = {
            "rate_of_change": rate,
            "rapid_change": rapid_change,
            "persistence": persistence,
            "stuck_value_indicator": stuck,
            "drift": drift,
        }

    return result
import numpy as np
import pandas as pd

from .schemas import CORE_VARIABLES, ContextConfig


def _safe_float(value):
    """Convert a value to float, returning None for missing values."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_feature(data, key):
    """Safely retrieve a numeric feature."""
    return _safe_float(data.get(key))


def _determine_temporal_pattern(
    current_data: dict,
    recent_data: pd.DataFrame,
    config: ContextConfig,
) -> str:
    """
    Determine a descriptive temporal pattern.

    This is evidence, not a fault classification.
    """

    if recent_data is None or recent_data.empty:
        return "insufficient_data"

    if "timestamp" in recent_data.columns:
        recent_data = recent_data.sort_values("timestamp")

    if len(recent_data) < 2:
        return "insufficient_data"

    # Focus on temperature for the primary temporal pattern.
    # Other variables are analyzed separately.
    values = pd.to_numeric(
        recent_data["temperature"],
        errors="coerce"
    ).dropna()

    if len(values) < 2:
        return "insufficient_data"

    diffs = values.diff().dropna()

    if len(diffs) == 0:
        return "stable"

    current_change = _get_feature(
        current_data,
        "temperature_diff_1"
    )

    if current_change is None:
        current_change = float(diffs.iloc[-1])

    recent_typical_change = float(
        np.median(np.abs(diffs.iloc[:-1]))
    ) if len(diffs) > 1 else 0.0

    # Sudden change if the current movement is much larger than
    # the recent typical movement.
    if abs(current_change) > max(
        3.0 * recent_typical_change,
        config.significant_change_thresholds["temperature"],
    ):
        return "sudden_change"

    # Check for a general trend.
    if len(values) >= 3:
        x = np.arange(len(values))
        slope = np.polyfit(x, values.to_numpy(), 1)[0]

        if abs(slope) >= config.drift_rate_thresholds["temperature"]:
            return "gradual_change"

    return "stable"


def calculate_temporal_context(
    current_data: dict,
    recent_data: pd.DataFrame,
    config: ContextConfig | None = None,
) -> dict:
    """
    Calculate temporal evidence for temperature, pressure and humidity.
    """

    config = config or ContextConfig()

    result = {}

    for variable in CORE_VARIABLES:
        current = _get_feature(current_data, variable)

        lag_1 = _get_feature(
            current_data,
            f"{variable}_lag_1"
        )

        lag_3 = _get_feature(
            current_data,
            f"{variable}_lag_3"
        )

        lag_5 = _get_feature(
            current_data,
            f"{variable}_lag_5"
        )

        rolling_mean_5 = _get_feature(
            current_data,
            f"{variable}_rolling_mean_5"
        )

        rolling_mean_15 = _get_feature(
            current_data,
            f"{variable}_rolling_mean_15"
        )

        rolling_std_5 = _get_feature(
            current_data,
            f"{variable}_rolling_std_5"
        )

        deviation = None
        z_score = None

        if (
            current is not None
            and rolling_mean_5 is not None
        ):
            deviation = current - rolling_mean_5

            if (
                rolling_std_5 is not None
                and rolling_std_5 > config.minimum_std
            ):
                z_score = deviation / rolling_std_5

        result[variable] = {
            "current": current,
            "lag_1": lag_1,
            "lag_3": lag_3,
            "lag_5": lag_5,
            "rolling_mean_5": rolling_mean_5,
            "rolling_mean_15": rolling_mean_15,
            "rolling_std_5": rolling_std_5,
            "deviation_from_rolling_mean_5": deviation,
            "rolling_z_score_5": z_score,
        }

    result["temporal_pattern"] = _determine_temporal_pattern(
        current_data,
        recent_data,
        config,
    )

    return result
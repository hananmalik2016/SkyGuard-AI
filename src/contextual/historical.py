import numpy as np
import pandas as pd

from .schemas import CORE_VARIABLES


def _historical_consistency(z_score):
    if z_score is None:
        return "insufficient_data"

    absolute = abs(z_score)

    if absolute < 1.0:
        return "high"

    if absolute < 2.0:
        return "medium"

    return "low"


def compare_historical_pattern(
    current_data: dict,
    historical_data: pd.DataFrame,
) -> dict:

    if (
        historical_data is None
        or historical_data.empty
    ):
        return {
            "available": False,
            "reason": "insufficient_historical_data",
        }

    historical = historical_data.copy()

    current_timestamp = pd.to_datetime(
        current_data["timestamp"],
        errors="coerce"
    )

    if pd.notna(current_timestamp):
        if "timestamp" in historical.columns:
            historical["timestamp"] = pd.to_datetime(
                historical["timestamp"],
                errors="coerce"
            )

            # Critical: prevent future-data leakage.
            historical = historical[
                historical["timestamp"] < current_timestamp
            ]

    if historical.empty:
        return {
            "available": False,
            "reason": "no_prior_historical_data",
        }

    # If station_id exists, only compare with the same station.
    station_id = current_data.get("station_id")

    if (
        station_id is not None
        and "station_id" in historical.columns
    ):
        historical = historical[
            historical["station_id"] == station_id
        ]

    if historical.empty:
        return {
            "available": False,
            "reason": "no_station_history",
        }

    result = {
        "available": True,
        "observation_count": len(historical),
    }

    current_hour = None

    if pd.notna(current_timestamp):
        current_hour = current_timestamp.hour

    same_hour_data = None

    if (
        current_hour is not None
        and "timestamp" in historical.columns
    ):
        same_hour_data = historical[
            historical["timestamp"].dt.hour == current_hour
        ]

        if same_hour_data.empty:
            same_hour_data = None

    for variable in CORE_VARIABLES:

        current_value = pd.to_numeric(
            current_data.get(variable),
            errors="coerce"
        )

        values = pd.to_numeric(
            historical[variable],
            errors="coerce"
        ).dropna()

        if values.empty or pd.isna(current_value):
            result[variable] = {
                "available": False,
                "reason": "insufficient_variable_history",
            }
            continue

        mean = float(values.mean())
        std = float(values.std(ddof=0))

        deviation = float(current_value - mean)

        if std > 1e-9:
            z_score = deviation / std
        else:
            z_score = None

        percentile = float(
            (values <= current_value).mean() * 100
        )

        variable_result = {
            "available": True,
            "historical_mean": mean,
            "historical_std": std,
            "deviation_from_mean": deviation,
            "z_score": z_score,
            "percentile": percentile,
            "consistency_with_baseline": (
                _historical_consistency(z_score)
            ),
        }

        # Optional same-time-of-day comparison.
        if (
            same_hour_data is not None
            and variable in same_hour_data.columns
        ):
            same_hour_values = pd.to_numeric(
                same_hour_data[variable],
                errors="coerce"
            ).dropna()

            if not same_hour_values.empty:
                same_hour_mean = float(
                    same_hour_values.mean()
                )

                variable_result[
                    "same_hour_mean"
                ] = same_hour_mean

                variable_result[
                    "same_hour_deviation"
                ] = float(
                    current_value - same_hour_mean
                )

        result[variable] = variable_result

    return result
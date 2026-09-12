import pandas as pd

from .schemas import (
    ContextConfig,
    validate_current_observation,
    validate_anomaly_result,
)

from .temporal import calculate_temporal_context
from .rate_persistence import analyze_rate_and_persistence
from .cross_variable import calculate_cross_variable_context
from .meteorological import evaluate_meteorological_plausibility
from .historical import compare_historical_pattern


def _prepare_dataframe(
    data,
    current_timestamp,
    station_id,
    filter_future=True,
):
    """
    Prepare recent/historical data safely.

    Only data from the same station is retained when station_id exists.
    """

    if data is None:
        return pd.DataFrame()

    if isinstance(data, list):
        data = pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        data = data.to_frame().T
    else:
        data = data.copy()

    if data.empty:
        return data

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

        data = data.dropna(
            subset=["timestamp"]
        )

        if filter_future and current_timestamp is not None:
            data = data[
                data["timestamp"] <= current_timestamp
            ]

    if (
        station_id is not None
        and "station_id" in data.columns
    ):
        data = data[
            data["station_id"] == station_id
        ]

    if "timestamp" in data.columns:
        data = data.sort_values("timestamp")

    return data


def analyze_context(
    current_data: dict,
    recent_data: pd.DataFrame | None,
    historical_data: pd.DataFrame | None,
    anomaly_result: dict,
    config: ContextConfig | None = None,
) -> dict:
    """
    Main Layer 5 entry point.

    Parameters
    ----------
    current_data:
        Current processed observation from Layer 3.

    recent_data:
        Previously available observations for the station.

    historical_data:
        Historical observations available before the current timestamp.

    anomaly_result:
        Result from Layer 4.

    config:
        Optional contextual analysis configuration.

    Returns
    -------
    dict
        Structured contextual evidence.

    IMPORTANT:
    This function does not make the final sensor-fault/weather-event
    classification.
    """

    config = config or ContextConfig()

    # ---------------------------------------------------------
    # 1. Validate basic interfaces
    # ---------------------------------------------------------

    missing_current = validate_current_observation(
        current_data
    )

    if missing_current:
        raise ValueError(
            f"Missing Layer 3 fields: {missing_current}"
        )

    missing_anomaly = validate_anomaly_result(
        anomaly_result
    )

    if missing_anomaly:
        raise ValueError(
            f"Missing Layer 4 fields: {missing_anomaly}"
        )

    # ---------------------------------------------------------
    # 2. Parse timestamp
    # ---------------------------------------------------------

    current_timestamp = pd.to_datetime(
        current_data["timestamp"],
        errors="coerce"
    )

    if pd.isna(current_timestamp):
        raise ValueError(
            "Invalid current observation timestamp."
        )

    station_id = current_data["station_id"]

    # ---------------------------------------------------------
    # 3. Prepare supporting data
    # ---------------------------------------------------------

    recent = _prepare_dataframe(
        recent_data,
        current_timestamp,
        station_id,
        filter_future=True,
    )

    historical = _prepare_dataframe(
        historical_data,
        current_timestamp,
        station_id,
        filter_future=True,
    )

    # ---------------------------------------------------------
    # 4. Calculate each contextual component
    # ---------------------------------------------------------

    temporal_context = calculate_temporal_context(
        current_data,
        recent,
        config,
    )

    rate_persistence = analyze_rate_and_persistence(
        current_data,
        recent,
        config,
    )

    cross_variable_context = calculate_cross_variable_context(
        current_data,
        recent,
        config,
    )

    meteorological_context = evaluate_meteorological_plausibility(
        temporal_context,
        rate_persistence,
        cross_variable_context,
        current_data,
    )

    historical_context = compare_historical_pattern(
        current_data,
        historical,
    )

    # ---------------------------------------------------------
    # 5. Structured Layer 5 output
    # ---------------------------------------------------------

    return {
        "station_id": station_id,
        "timestamp": current_timestamp.isoformat(),

        "anomaly_context": {
            "anomaly_score": anomaly_result.get(
                "anomaly_score"
            ),
            "anomaly_status": anomaly_result.get(
                "anomaly_status"
            ),
            "statistical_score": anomaly_result.get(
                "statistical_score"
            ),
            "ml_score": anomaly_result.get(
                "ml_score"
            ),
            "model": anomaly_result.get(
                "model"
            ),
            "model_version": anomaly_result.get(
                "model_version"
            ),
        },

        "temporal_context": temporal_context,

        "rate_persistence": rate_persistence,

        "cross_variable_context": cross_variable_context,

        "meteorological_context": meteorological_context,

        "historical_context": historical_context,
    }
import pandas as pd

from .lag_features import create_lag_features
from .rolling_features import create_rolling_features
from .temporal_features import create_temporal_features


SENSOR_COLUMNS = [
    "temperature",
    "pressure",
    "humidity"
]

REQUIRED_COLUMNS = [
    "station_id",
    "timestamp",
    "temperature",
    "pressure",
    "humidity"
]


def validate_input(df: pd.DataFrame) -> None:
    """Validate the input dataframe."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the input data for feature engineering."""

    result = df.copy()

    result["timestamp"] = pd.to_datetime(result["timestamp"])

    result = (
        result
        .sort_values(["station_id", "timestamp"])
        .reset_index(drop=True)
    )

    return result


def create_difference_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Create one-step difference features."""

    result = df.copy()

    for sensor in SENSOR_COLUMNS:

        result[f"{sensor}_diff_1"] = (
            result.groupby("station_id")[sensor]
            .diff(1)
        )

        result[f"{sensor}_abs_diff"] = (
            result[f"{sensor}_diff_1"].abs()
        )

    return result


def create_rate_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Create rate-of-change features using actual elapsed time."""

    result = df.copy()

    time_difference = (
        result.groupby("station_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    time_difference = time_difference.replace(0, pd.NA)

    for sensor in SENSOR_COLUMNS:

        result[f"{sensor}_rate"] = (
            result[f"{sensor}_diff_1"]
            / time_difference
        )

    return result


def preserve_quality_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Preserve or create data-quality indicators."""

    result = df.copy()

    if "is_imputed" not in result.columns:
        result["is_imputed"] = False

    if "gap_detected" not in result.columns:
        result["gap_detected"] = False

    if "time_diff" not in result.columns:
        result["time_diff"] = (
            result.groupby("station_id")["timestamp"]
            .diff()
            .dt.total_seconds()
        )

    return result


def create_all_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run the complete Layer 3 feature-engineering pipeline.

    Layer 3 creates descriptive features only.
    It does not classify observations as anomalies.
    """

    validate_input(df)

    result = prepare_dataframe(df)

    result = create_lag_features(result)

    result = create_difference_features(result)

    result = create_rate_features(result)

    result = create_rolling_features(result)

    result = create_temporal_features(result)

    result = preserve_quality_features(result)

    result = (
        result
        .sort_values(["station_id", "timestamp"])
        .reset_index(drop=True)
    )

    return result
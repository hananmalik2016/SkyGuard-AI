import pandas as pd


SENSOR_COLUMNS = [
    "temperature",
    "pressure",
    "humidity"
]

ROLLING_WINDOWS = [5, 15]


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create historical rolling features for each sensor.

    The current observation is excluded from the rolling calculation
    to prevent future/current-value leakage.
    """

    required_columns = ["station_id", "timestamp"] + SENSOR_COLUMNS

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(result["timestamp"])

    result = result.sort_values(
        ["station_id", "timestamp"]
    ).reset_index(drop=True)

    for sensor in SENSOR_COLUMNS:
        grouped = result.groupby("station_id")[sensor]

        historical = grouped.shift(1)

        for window in ROLLING_WINDOWS:
            result[f"{sensor}_rolling_mean_{window}"] = (
                historical.groupby(result["station_id"])
                .transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            )

        result[f"{sensor}_rolling_std_5"] = (
            historical.groupby(result["station_id"])
            .transform(
                lambda x: x.rolling(5, min_periods=2).std()
            )
        )

        result[f"{sensor}_rolling_min_15"] = (
            historical.groupby(result["station_id"])
            .transform(
                lambda x: x.rolling(15, min_periods=1).min()
            )
        )

        result[f"{sensor}_rolling_max_15"] = (
            historical.groupby(result["station_id"])
            .transform(
                lambda x: x.rolling(15, min_periods=1).max()
            )
        )

    return result
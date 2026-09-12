import pandas as pd


SENSOR_COLUMNS = [
    "temperature",
    "pressure",
    "humidity"
]

DEFAULT_LAGS = [1, 3, 5]


def create_lag_features(df: pd.DataFrame, lags=None) -> pd.DataFrame:
    """
    Create lag features for each sensor independently for each AWS station.

    A lag feature contains the value of a sensor from a previous observation.
    """

    if lags is None:
        lags = DEFAULT_LAGS

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
        for lag in lags:
            result[f"{sensor}_lag_{lag}"] = (
                result.groupby("station_id")[sensor]
                .shift(lag)
            )

    return result
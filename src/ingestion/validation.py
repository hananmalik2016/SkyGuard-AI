import pandas as pd

from .ingestion import REQUIRED_COLUMNS


def validate_data(df):
    """Validate the structure and basic values of raw weather data."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["station_id"].isna().any():
        raise ValueError("Missing station_id values")

    if df["timestamp"].isna().any():
        raise ValueError("Missing timestamp values")

    for column in ["temperature", "pressure", "humidity"]:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"{column} must contain numeric values"
            )

    return True
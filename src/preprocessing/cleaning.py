import pandas as pd


def clean_data(df):
    """Clean and prepare raw weather data."""

    result = df.copy()

    # Convert timestamp to datetime
    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce"
    )

    # Remove completely invalid timestamps
    result = result.dropna(
        subset=["timestamp"]
    )

    # Remove duplicate station/timestamp records
    result = result.drop_duplicates(
        subset=["station_id", "timestamp"]
    )

    # Sort chronologically for each station
    result = result.sort_values(
        ["station_id", "timestamp"]
    ).reset_index(drop=True)

    # Detect time gaps
    result["time_diff"] = (
        result.groupby("station_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    result["gap_detected"] = (
        result["time_diff"] > 600
    )

    # Track whether values were imputed
    result["is_imputed"] = False

    return result
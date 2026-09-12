import numpy as np
import pandas as pd


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the timestamp column.
    """

    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")

    result = df.copy()

    result["timestamp"] = pd.to_datetime(result["timestamp"])

    hour = (
        result["timestamp"].dt.hour
        + result["timestamp"].dt.minute / 60
        + result["timestamp"].dt.second / 3600
    )

    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return result
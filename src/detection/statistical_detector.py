import numpy as np
import pandas as pd

from .schemas import DetectionConfig


SENSORS = [
    "temperature",
    "pressure",
    "humidity",
]


class StatisticalDetector:
    """
    Lightweight statistical anomaly detector.

    Uses:
    - previous observation
    - rolling statistics
    - absolute differences
    - rate of change
    """

    def __init__(self, config: DetectionConfig):
        self.config = config

    def _z_score(self, value, mean, std):
        if pd.isna(value) or pd.isna(mean):
            return 0.0

        if pd.isna(std) or std < 1e-9:
            return 0.0

        return abs((value - mean) / std)

    def _z_to_score(self, z):
        """
        Convert z-score into a bounded 0-1 anomaly score.
        """

        if z <= 0:
            return 0.0

        threshold = self.config.z_threshold

        return float(
            np.clip(
                z / (z + threshold),
                0.0,
                1.0,
            )
        )

    def detect(self, row: pd.Series) -> dict:

        sensor_scores = {}

        for sensor in SENSORS:

            rolling_mean = row.get(
                f"{sensor}_rolling_mean_5",
                np.nan,
            )

            rolling_std = row.get(
                f"{sensor}_rolling_std_5",
                np.nan,
            )

            value = row.get(sensor, np.nan)

            z = self._z_score(
                value,
                rolling_mean,
                rolling_std,
            )

            z_score = self._z_to_score(z)

            # Sudden-change signal
            abs_diff = row.get(
                f"{sensor}_abs_diff",
                0.0,
            )

            if pd.isna(abs_diff):
                abs_diff = 0.0

            # Rate signal
            rate = row.get(
                f"{sensor}_rate",
                0.0,
            )

            if pd.isna(rate):
                rate = 0.0

            # Normalize change signal using conservative prototype limits
            change_score = min(
                abs(float(abs_diff)) / 5.0,
                1.0,
            )

            rate_limits = {
                "temperature": 1.0,
                "pressure": 0.5,
                "humidity": 3.0,
            }

            rate_score = min(
                abs(float(rate))
                / rate_limits[sensor],
                1.0,
            )

            # Combine three statistical signals
            final_sensor_score = (
                0.50 * z_score
                + 0.30 * change_score
                + 0.20 * rate_score
            )

            sensor_scores[sensor] = float(
                np.clip(
                    final_sensor_score,
                    0.0,
                    1.0,
                )
            )

        # Strongest sensor anomaly
        overall_score = max(
            sensor_scores.values()
        )

        return {
            "statistical_score": overall_score,
            "sensor_scores": sensor_scores,
        }
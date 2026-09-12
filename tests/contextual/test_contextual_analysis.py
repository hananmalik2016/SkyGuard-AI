import unittest

import pandas as pd

from src.contextual import analyze_context


class TestContextualAnalysis(unittest.TestCase):

    def setUp(self):

        self.anomaly_result = {
            "station_id": "AWS_001",
            "timestamp": "2026-09-12T14:03:00",
            "temperature": 41.8,
            "pressure": 1008.9,
            "humidity": 71.2,
            "anomaly_score": 0.94,
            "anomaly_status": "anomalous",
            "statistical_score": 0.97,
            "ml_score": 0.91,
            "model": "isolation_forest",
            "model_version": "1.0",
        }

    def make_current(
        self,
        temperature=28.5,
        pressure=1008.0,
        humidity=65.0,
        temperature_change=0.1,
        pressure_change=0.1,
        humidity_change=0.2,
    ):
        return {
            "station_id": "AWS_001",
            "timestamp": "2026-09-12T14:03:00",

            "temperature": temperature,
            "pressure": pressure,
            "humidity": humidity,

            "temperature_lag_1": (
                temperature - temperature_change
            ),
            "temperature_lag_3": temperature,
            "temperature_lag_5": temperature,

            "pressure_lag_1": (
                pressure - pressure_change
            ),
            "pressure_lag_3": pressure,
            "pressure_lag_5": pressure,

            "humidity_lag_1": (
                humidity - humidity_change
            ),
            "humidity_lag_3": humidity,
            "humidity_lag_5": humidity,

            "temperature_diff_1": temperature_change,
            "temperature_abs_diff": abs(
                temperature_change
            ),

            "pressure_diff_1": pressure_change,
            "pressure_abs_diff": abs(
                pressure_change
            ),

            "humidity_diff_1": humidity_change,
            "humidity_abs_diff": abs(
                humidity_change
            ),

            "temperature_rate": temperature_change,
            "pressure_rate": pressure_change,
            "humidity_rate": humidity_change,

            "temperature_rolling_mean_5": temperature,
            "temperature_rolling_mean_15": temperature,
            "temperature_rolling_std_5": 0.5,
            "temperature_rolling_min_15": temperature,
            "temperature_rolling_max_15": temperature,

            "pressure_rolling_mean_5": pressure,
            "pressure_rolling_mean_15": pressure,
            "pressure_rolling_std_5": 0.5,
            "pressure_rolling_min_15": pressure,
            "pressure_rolling_max_15": pressure,

            "humidity_rolling_mean_5": humidity,
            "humidity_rolling_mean_15": humidity,
            "humidity_rolling_std_5": 1.0,
            "humidity_rolling_min_15": humidity,
            "humidity_rolling_max_15": humidity,

            "hour_sin": 0.0,
            "hour_cos": 1.0,

            "is_imputed": False,
            "gap_detected": False,
            "time_diff": 60.0,

            "is_anomaly": False,
            "anomaly_type": None,
        }

    def make_recent(
        self,
        temperatures,
        pressures,
        humidities,
    ):
        timestamps = pd.date_range(
            "2026-09-12 13:58:00",
            periods=len(temperatures),
            freq="1min",
        )

        return pd.DataFrame({
            "station_id": ["AWS_001"] * len(timestamps),
            "timestamp": timestamps,
            "temperature": temperatures,
            "pressure": pressures,
            "humidity": humidities,
        })

    def test_normal_data(self):

        current = self.make_current()

        recent = self.make_recent(
            [28.1, 28.2, 28.3, 28.4, 28.5],
            [1007.8, 1007.9, 1008.0, 1008.0, 1008.0],
            [64.5, 64.7, 64.8, 64.9, 65.0],
        )

        result = analyze_context(
            current,
            recent,
            recent,
            self.anomaly_result,
        )

        self.assertEqual(
            result["station_id"],
            "AWS_001"
        )

        self.assertIn(
            "temporal_context",
            result
        )

    def test_temperature_spike(self):

        current = self.make_current(
            temperature=42.0,
            temperature_change=13.5,
        )

        recent = self.make_recent(
            [28.0, 28.2, 28.3, 28.5, 28.5],
            [1008.0] * 5,
            [65.0] * 5,
        )

        result = analyze_context(
            current,
            recent,
            recent,
            self.anomaly_result,
        )

        self.assertTrue(
            result["rate_persistence"]["temperature"][
                "rapid_change"
            ]
        )

        self.assertEqual(
            result["cross_variable_context"][
                "overall_consistency"
            ],
            "low",
        )

    def test_pressure_spike(self):

        current = self.make_current(
            pressure=1020.0,
            pressure_change=12.0,
        )

        recent = self.make_recent(
            [28.0] * 5,
            [1008.0] * 5,
            [65.0] * 5,
        )

        result = analyze_context(
            current,
            recent,
            recent,
            self.anomaly_result,
        )

        self.assertTrue(
            result["rate_persistence"]["pressure"][
                "rapid_change"
            ]
        )

    def test_humidity_anomaly(self):

        current = self.make_current(
            humidity=95.0,
            humidity_change=25.0,
        )

        recent = self.make_recent(
            [28.0] * 5,
            [1008.0] * 5,
            [65.0] * 5,
        )

        result = analyze_context(
            current,
            recent,
            recent,
            self.anomaly_result,
        )

        self.assertTrue(
            result["rate_persistence"]["humidity"][
                "rapid_change"
            ]
        )

    def test_stuck_sensor(self):

        current = self.make_current(
            temperature=30.0,
            temperature_change=0.0,
        )

        recent = self.make_recent(
            [30.0, 30.0, 30.0, 30.0, 30.0],
            [1008.0] * 5,
            [65.0] * 5,
        )

        result = analyze_context(
            current,
            recent,
            recent,
            self.anomaly_result,
        )

        self.assertTrue(
            result["rate_persistence"]["temperature"][
                "stuck_value_indicator"
            ]
        )

    def test_gradual_drift(self):

        current = self.make_current(
            temperature=30.0,
            temperature_change=0.5,
        )

        recent = self.make_recent(
            [28.0, 28.5, 29.0, 29.5, 30.0],
            [1008.0] * 5,
            [65.0] * 5,
        )

        result = analyze_context(
            current,
            recent,
            recent,
            self.anomaly_result,
        )

        drift = result["rate_persistence"][
            "temperature"
        ]["drift"]

        self.assertIsNotNone(drift)

    def test_cross_variable_consistent_change(self):

        current = self.make_current(
            temperature=34.0,
            pressure=1004.0,
            humidity=75.0,
            temperature_change=5.0,
            pressure_change=-4.0,
            humidity_change=10.0,
        )

        recent = self.make_recent(
            [28.0, 28.1, 28.2, 28.3, 29.0],
            [1008.0] * 5,
            [65.0] * 5,
        )

        result = analyze_context(
            current,
            recent,
            recent,
            self.anomaly_result,
        )

        self.assertEqual(
            result["cross_variable_context"][
                "number_of_changed_variables"
            ],
            3,
        )

    def test_missing_historical_data(self):

        current = self.make_current()

        recent = self.make_recent(
            [28.1, 28.2, 28.3, 28.4, 28.5],
            [1007.8, 1007.9, 1008.0, 1008.0, 1008.0],
            [64.5, 64.7, 64.8, 64.9, 65.0],
        )

        result = analyze_context(
            current,
            recent,
            None,
            self.anomaly_result,
        )

        self.assertFalse(
            result["historical_context"]["available"]
        )


if __name__ == "__main__":
    unittest.main()
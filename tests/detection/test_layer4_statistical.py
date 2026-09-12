import unittest
import pandas as pd

from src.detection.schemas import DetectionConfig
from src.detection.statistical_detector import StatisticalDetector


class TestStatisticalDetector(unittest.TestCase):

    def setUp(self):
        self.config = DetectionConfig(
            z_threshold=3.0,
            suspicious_threshold=0.50,
            anomalous_threshold=0.80,
        )

        self.detector = StatisticalDetector(self.config)

    def make_row(
        self,
        temperature=25.0,
        pressure=1013.0,
        humidity=60.0,
        temperature_mean=25.0,
        pressure_mean=1013.0,
        humidity_mean=60.0,
        temperature_std=1.0,
        pressure_std=1.0,
        humidity_std=2.0,
        temperature_diff=0.0,
        pressure_diff=0.0,
        humidity_diff=0.0,
        temperature_rate=0.0,
        pressure_rate=0.0,
        humidity_rate=0.0,
    ):
        return pd.Series({
            "station_id": "AWS_001",
            "timestamp": "2026-09-12T12:00:00",

            "temperature": temperature,
            "pressure": pressure,
            "humidity": humidity,

            "temperature_rolling_mean_5": temperature_mean,
            "temperature_rolling_std_5": temperature_std,

            "pressure_rolling_mean_5": pressure_mean,
            "pressure_rolling_std_5": pressure_std,

            "humidity_rolling_mean_5": humidity_mean,
            "humidity_rolling_std_5": humidity_std,

            "temperature_abs_diff": abs(temperature_diff),
            "pressure_abs_diff": abs(pressure_diff),
            "humidity_abs_diff": abs(humidity_diff),

            "temperature_rate": temperature_rate,
            "pressure_rate": pressure_rate,
            "humidity_rate": humidity_rate,
        })

    # --------------------------------------------------
    # 1. Normal observation
    # --------------------------------------------------

    def test_normal_observation(self):
        row = self.make_row()

        result = self.detector.detect(row)

        self.assertIn("statistical_score", result)
        self.assertIn("sensor_scores", result)

        self.assertGreaterEqual(
            result["statistical_score"], 0.0
        )

        self.assertLessEqual(
            result["statistical_score"], 1.0
        )

        self.assertLess(
            result["statistical_score"], 0.50
        )

    # --------------------------------------------------
    # 2. Temperature spike
    # --------------------------------------------------

    def test_temperature_spike(self):
        row = self.make_row(
            temperature=40.0,
            temperature_mean=25.0,
            temperature_std=1.0,
            temperature_diff=15.0,
            temperature_rate=2.0,
        )

        result = self.detector.detect(row)

        self.assertGreater(
            result["sensor_scores"]["temperature"],
            0.80,
        )

        self.assertGreater(
            result["statistical_score"],
            0.80,
        )

    # --------------------------------------------------
    # 3. Pressure spike
    # --------------------------------------------------

    def test_pressure_spike(self):
        row = self.make_row(
            pressure=1023.0,
            pressure_mean=1013.0,
            pressure_std=1.0,
            pressure_diff=10.0,
            pressure_rate=1.0,
        )

        result = self.detector.detect(row)

        self.assertGreater(
            result["sensor_scores"]["pressure"],
            0.80,
        )

        self.assertGreater(
            result["statistical_score"],
            0.80,
        )

    # --------------------------------------------------
    # 4. Humidity spike
    # --------------------------------------------------

    def test_humidity_spike(self):
        row = self.make_row(
            humidity=90.0,
            humidity_mean=60.0,
            humidity_std=2.0,
            humidity_diff=30.0,
            humidity_rate=10.0,
        )

        result = self.detector.detect(row)

        self.assertGreater(
            result["sensor_scores"]["humidity"],
            0.80,
        )

        self.assertGreater(
            result["statistical_score"],
            0.80,
        )

    # --------------------------------------------------
    # 5. Gradual drift
    # --------------------------------------------------

    def test_gradual_drift_is_less_extreme_than_spike(self):
        drift = self.make_row(
            temperature=28.0,
            temperature_mean=25.0,
            temperature_std=1.5,
            temperature_diff=0.5,
            temperature_rate=0.1,
        )

        spike = self.make_row(
            temperature=40.0,
            temperature_mean=25.0,
            temperature_std=1.0,
            temperature_diff=15.0,
            temperature_rate=2.0,
        )

        drift_result = self.detector.detect(drift)
        spike_result = self.detector.detect(spike)

        self.assertLess(
            drift_result["statistical_score"],
            spike_result["statistical_score"],
        )

    # --------------------------------------------------
    # 6. Stuck sensor
    # --------------------------------------------------

    def test_stuck_sensor_with_zero_change(self):
        row = self.make_row(
            temperature=25.0,
            temperature_mean=25.0,
            temperature_std=0.0,
            temperature_diff=0.0,
            temperature_rate=0.0,
        )

        result = self.detector.detect(row)

        self.assertEqual(
            result["sensor_scores"]["temperature"],
            0.0,
        )

    # --------------------------------------------------
    # 7. Multivariate anomaly
    # --------------------------------------------------

    def test_multivariate_anomaly(self):
        row = self.make_row(
            temperature=40.0,
            pressure=1023.0,
            humidity=90.0,

            temperature_mean=25.0,
            pressure_mean=1013.0,
            humidity_mean=60.0,

            temperature_std=1.0,
            pressure_std=1.0,
            humidity_std=2.0,

            temperature_diff=15.0,
            pressure_diff=10.0,
            humidity_diff=30.0,

            temperature_rate=2.0,
            pressure_rate=1.0,
            humidity_rate=10.0,
        )

        result = self.detector.detect(row)

        self.assertGreater(
            result["sensor_scores"]["temperature"],
            0.80,
        )

        self.assertGreater(
            result["sensor_scores"]["pressure"],
            0.80,
        )

        self.assertGreater(
            result["sensor_scores"]["humidity"],
            0.80,
        )

        self.assertGreater(
            result["statistical_score"],
            0.80,
        )

    # --------------------------------------------------
    # 8. Genuine weather transition candidate
    # --------------------------------------------------

    def test_coordinated_weather_transition(self):
        row = self.make_row(
            temperature=32.0,
            pressure=1008.0,
            humidity=78.0,

            temperature_mean=27.0,
            pressure_mean=1013.0,
            humidity_mean=60.0,

            temperature_std=2.0,
            pressure_std=2.0,
            humidity_std=5.0,

            temperature_diff=5.0,
            pressure_diff=-5.0,
            humidity_diff=18.0,

            temperature_rate=0.5,
            pressure_rate=-0.3,
            humidity_rate=2.0,
        )

        result = self.detector.detect(row)

        # Layer 4 should detect unusual behavior.
        # It must NOT decide whether this is weather
        # or a sensor fault.
        self.assertGreaterEqual(
            result["statistical_score"],
            0.0,
        )

        self.assertLessEqual(
            result["statistical_score"],
            1.0,
        )

    # --------------------------------------------------
    # 9. Missing statistical features
    # --------------------------------------------------

    def test_missing_statistical_features_do_not_crash(self):
        row = pd.Series({
            "station_id": "AWS_001",
            "timestamp": "2026-09-12T12:00:00",

            "temperature": 25.0,
            "pressure": 1013.0,
            "humidity": 60.0,
        })

        result = self.detector.detect(row)

        self.assertIn(
            "statistical_score",
            result,
        )

        self.assertGreaterEqual(
            result["statistical_score"], 0.0
        )

        self.assertLessEqual(
            result["statistical_score"], 1.0
        )

    # --------------------------------------------------
    # 10. Multiple stations
    # --------------------------------------------------

    def test_station_id_is_preserved(self):
        row = self.make_row()

        row["station_id"] = "AWS_999"

        result = self.detector.detect(row)

        # The detector currently doesn't return
        # station_id, so simply verify it can process
        # different station identifiers.
        self.assertIn(
            "statistical_score",
            result,
        )

    # --------------------------------------------------
    # 11. Output contract
    # --------------------------------------------------

    def test_output_contract(self):
        row = self.make_row()

        result = self.detector.detect(row)

        self.assertIsInstance(result, dict)

        self.assertIn(
            "statistical_score",
            result,
        )

        self.assertIn(
            "sensor_scores",
            result,
        )

        self.assertIsInstance(
            result["statistical_score"],
            float,
        )

        self.assertIsInstance(
            result["sensor_scores"],
            dict,
        )

        for sensor in [
            "temperature",
            "pressure",
            "humidity",
        ]:
            self.assertIn(
                sensor,
                result["sensor_scores"],
            )

            self.assertGreaterEqual(
                result["sensor_scores"][sensor],
                0.0,
            )

            self.assertLessEqual(
                result["sensor_scores"][sensor],
                1.0,
            )

    # --------------------------------------------------
    # 12. Score monotonicity
    # --------------------------------------------------

    def test_larger_temperature_deviation_increases_score(self):
        small = self.make_row(
            temperature=27.0,
            temperature_mean=25.0,
            temperature_std=1.0,
            temperature_diff=2.0,
            temperature_rate=0.2,
        )

        large = self.make_row(
            temperature=35.0,
            temperature_mean=25.0,
            temperature_std=1.0,
            temperature_diff=10.0,
            temperature_rate=1.5,
        )

        small_result = self.detector.detect(small)
        large_result = self.detector.detect(large)

        self.assertGreater(
            large_result["statistical_score"],
            small_result["statistical_score"],
        )


if __name__ == "__main__":
    unittest.main()

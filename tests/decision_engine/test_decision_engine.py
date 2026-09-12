import unittest

from src.decision_engine import make_context_aware_decision


class TestDecisionEngine(unittest.TestCase):

    def test_normal_observation(self):

        anomaly = {
            "anomaly_score": 0.10,
            "anomaly_status": "normal",
        }

        context = {}

        result = make_context_aware_decision(
            anomaly,
            context,
        )

        self.assertEqual(
            result["classification"],
            "NORMAL",
        )


    def test_isolated_sensor_spike(self):

        anomaly = {
            "anomaly_score": 0.95,
            "anomaly_status": "anomalous",
        }

        context = {
            "temporal_context": {
                "temporal_pattern": "sudden_change",
            },

            "rate_persistence": {
                "temperature": {
                    "rapid_change": True,
                    "persistence": False,
                    "stuck_value_indicator": False,
                }
            },

            "cross_variable_context": {
                "overall_consistency": "low",
                "number_of_changed_variables": 1,
            },

            "meteorological_context": {
                "plausibility": "low",
                "persistent_change_count": 0,
            },

            "historical_context": {
                "available": True,

                "temperature": {
                    "consistency_with_baseline": "low",
                }
            },
        }

        result = make_context_aware_decision(
            anomaly,
            context,
        )

        self.assertEqual(
            result["classification"],
            "SENSOR_FAULT",
        )


    def test_genuine_weather_event(self):

        anomaly = {
            "anomaly_score": 0.90,
            "anomaly_status": "anomalous",
        }

        context = {
            "cross_variable_context": {
                "overall_consistency": "high",
                "number_of_changed_variables": 3,
            },

            "rate_persistence": {
                "temperature": {
                    "persistence": True,
                },
                "pressure": {
                    "persistence": True,
                },
                "humidity": {
                    "persistence": True,
                },
            },

            "meteorological_context": {
                "plausibility": "high",
                "persistent_change_count": 3,
            },

            "historical_context": {
                "available": True,

                "temperature": {
                    "consistency_with_baseline": "high",
                },

                "pressure": {
                    "consistency_with_baseline": "high",
                },

                "humidity": {
                    "consistency_with_baseline": "high",
                },
            },
        }

        result = make_context_aware_decision(
            anomaly,
            context,
        )

        self.assertEqual(
            result["classification"],
            "GENUINE_METEOROLOGICAL_EVENT",
        )


    def test_communication_issue(self):

        anomaly = {
            "anomaly_score": 0.90,
            "anomaly_status": "anomalous",
        }

        context = {
            "data_quality_context": {
                "communication_issue": True,
            }
        }

        result = make_context_aware_decision(
            anomaly,
            context,
        )

        self.assertEqual(
            result["classification"],
            "DATA_COMMUNICATION_ISSUE",
        )


    def test_stuck_sensor(self):

        anomaly = {
            "anomaly_score": 0.85,
            "anomaly_status": "anomalous",
        }

        context = {
            "rate_persistence": {
                "temperature": {
                    "rapid_change": False,
                    "persistence": False,
                    "stuck_value_indicator": True,
                }
            },

            "cross_variable_context": {
                "overall_consistency": "low",
                "number_of_changed_variables": 1,
            },

            "meteorological_context": {
                "plausibility": "low",
                "persistent_change_count": 0,
            },
        }

        result = make_context_aware_decision(
            anomaly,
            context,
        )

        self.assertEqual(
            result["classification"],
            "SENSOR_FAULT",
        )


    def test_conflicting_evidence(self):

        anomaly = {
            "anomaly_score": 0.95,
            "anomaly_status": "anomalous",
        }

        context = {
            "temporal_context": {
                "temporal_pattern": "sudden_change",
            },

            "rate_persistence": {
                "temperature": {
                    "rapid_change": True,
                    "persistence": True,
                },

                "pressure": {
                    "rapid_change": False,
                    "persistence": True,
                },

                "humidity": {
                    "rapid_change": False,
                    "persistence": True,
                },
            },

            "cross_variable_context": {
                "overall_consistency": "high",
                "number_of_changed_variables": 3,
            },

            "meteorological_context": {
                "plausibility": "high",
                "persistent_change_count": 3,
            },
        }

        result = make_context_aware_decision(
            anomaly,
            context,
        )

        self.assertEqual(
            result["classification"],
            "UNCERTAIN",
        )


    def test_insufficient_context(self):

        anomaly = {
            "anomaly_score": 0.95,
            "anomaly_status": "anomalous",
        }

        context = {}

        result = make_context_aware_decision(
            anomaly,
            context,
        )

        self.assertEqual(
            result["classification"],
            "UNCERTAIN",
        )


    def test_low_anomaly_score(self):

        anomaly = {
            "anomaly_score": 0.20,
            "anomaly_status": "normal",
        }

        context = {
            "temporal_context": {
                "temporal_pattern": "stable",
            }
        }

        result = make_context_aware_decision(
            anomaly,
            context,
        )

        self.assertEqual(
            result["classification"],
            "NORMAL",
        )


if __name__ == "__main__":
    unittest.main()
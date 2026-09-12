import pandas as pd

from .schemas import DetectionConfig
from .statistical_detector import StatisticalDetector
from .isolation_forest import IsolationForestDetector
from .ensemble import (
    combine_scores,
    classify_score,
)


class AnomalyDetectionEngine:

    def __init__(
        self,
        config=None,
    ):

        self.config = (
            config
            or DetectionConfig()
        )

        self.statistical_detector = (
            StatisticalDetector(
                self.config
            )
        )

        self.ml_detector = (
            IsolationForestDetector(
                n_estimators=self.config.n_estimators,
                contamination=self.config.contamination,
                random_state=self.config.random_state,
            )
        )

        self.is_trained = False

    def train(
        self,
        historical_features: pd.DataFrame,
    ):

        if len(historical_features) < (
            self.config.minimum_history
        ):
            raise ValueError(
                "Not enough historical data "
                "to train anomaly detector."
            )

        self.ml_detector.fit(
            historical_features
        )

        self.is_trained = True

    def predict(
        self,
        current_row: pd.Series,
    ):

        if not self.is_trained:
            raise RuntimeError(
                "Detection engine has not been trained."
            )

        # -------------------------
        # Statistical detector
        # -------------------------

        statistical_result = (
            self.statistical_detector.detect(
                current_row
            )
        )

        statistical_score = (
            statistical_result[
                "statistical_score"
            ]
        )

        # -------------------------
        # ML detector
        # -------------------------

        current_df = pd.DataFrame(
            [current_row]
        )

        ml_score = float(
            self.ml_detector.predict_score(
                current_df
            )[0]
        )

        # -------------------------
        # Ensemble
        # -------------------------

        anomaly_score = combine_scores(
            statistical_score,
            ml_score,
            self.config.statistical_weight,
            self.config.ml_weight,
        )

        anomaly_status = classify_score(
            anomaly_score,
            self.config.suspicious_threshold,
            self.config.anomalous_threshold,
        )

        timestamp = pd.to_datetime(
            current_row["timestamp"]
        ).isoformat()

        return {
            "station_id": current_row[
                "station_id"
            ],

            "timestamp": timestamp,

            "anomaly_score": anomaly_score,

            "anomaly_status": anomaly_status,

            "statistical_score":
                statistical_score,

            "ml_score":
                ml_score,

            "sensor_scores":
                statistical_result[
                    "sensor_scores"
                ],

            "model":
                "hybrid_statistical_isolation_forest",

            "model_version":
                "1.0",
        }
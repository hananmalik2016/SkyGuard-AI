import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


EXCLUDED_COLUMNS = {
    "station_id",
    "timestamp",

    # Labels / ground truth
    "is_anomaly",
    "anomaly_type",
}


class IsolationForestDetector:

    def __init__(
        self,
        n_estimators=200,
        contamination="auto",
        random_state=42,
    ):

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

        self.feature_columns = None
        self.feature_medians = None

        self.score_low = None
        self.score_high = None

    def _prepare_features(
        self,
        df: pd.DataFrame,
        training=False,
    ):

        X = df.copy()

        # Remove non-feature columns
        columns_to_drop = [
            column
            for column in EXCLUDED_COLUMNS
            if column in X.columns
        ]

        X = X.drop(
            columns=columns_to_drop,
            errors="ignore",
        )

        # Keep numeric features only
        X = X.select_dtypes(
            include=[np.number]
        )

        # Replace infinite values
        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if training:

            self.feature_columns = list(
                X.columns
            )

            self.feature_medians = (
                X.median()
            )

        else:

            # Guarantee same columns as training
            for column in self.feature_columns:

                if column not in X.columns:
                    X[column] = np.nan

            X = X[
                self.feature_columns
            ]

        # Use training medians
        X = X.fillna(
            self.feature_medians
        )

        return X

    def fit(self, df: pd.DataFrame):

        X = self._prepare_features(
            df,
            training=True,
        )

        self.model.fit(X)

        # Learn score range from training data
        training_decision = (
            self.model.decision_function(X)
        )

        self.score_low = float(
            np.percentile(
                training_decision,
                5,
            )
        )

        self.score_high = float(
            np.percentile(
                training_decision,
                95,
            )
        )

        return self

    def predict_score(
        self,
        df: pd.DataFrame,
    ):

        X = self._prepare_features(
            df,
            training=False,
        )

        decision = (
            self.model.decision_function(X)
        )

        # Isolation Forest:
        # higher decision value = more normal
        #
        # Convert it so:
        # higher score = more anomalous

        denominator = (
            self.score_high
            - self.score_low
        )

        if denominator <= 1e-9:
            return np.zeros(
                len(df)
            )

        anomaly_score = (
            self.score_high - decision
        ) / denominator

        anomaly_score = np.clip(
            anomaly_score,
            0.0,
            1.0,
        )

        return anomaly_score
import numpy as np


def combine_scores(
    statistical_score,
    ml_score,
    statistical_weight=0.40,
    ml_weight=0.60,
):

    total_weight = (
        statistical_weight
        + ml_weight
    )

    score = (
        statistical_weight
        * statistical_score
        +
        ml_weight
        * ml_score
    ) / total_weight

    return float(
        np.clip(score, 0.0, 1.0)
    )


def classify_score(
    anomaly_score,
    suspicious_threshold=0.50,
    anomalous_threshold=0.80,
):

    if anomaly_score >= anomalous_threshold:

        return "anomalous"

    elif anomaly_score >= suspicious_threshold:

        return "suspicious"

    return "normal"
from dataclasses import dataclass


@dataclass
class DetectionConfig:
    # Final anomaly thresholds
    suspicious_threshold: float = 0.50
    anomalous_threshold: float = 0.80

    # Statistical detector
    z_threshold: float = 3.0

    # Ensemble weights
    statistical_weight: float = 0.40
    ml_weight: float = 0.60

    # Isolation Forest
    n_estimators: int = 200
    contamination: str = "auto"
    random_state: int = 42

    # Minimum historical observations
    minimum_history: int = 10
from dataclasses import dataclass


@dataclass
class DecisionConfig:
    # Anomaly threshold
    anomaly_threshold: float = 0.50

    # Minimum evidence required for a strong decision
    decision_threshold: float = 0.60

    # If two competing decisions are this close,
    # treat the result as conflicting evidence.
    conflict_margin: float = 0.10

    # Weights for sensor-fault evidence
    sensor_fault_weights: dict = None

    # Weights for genuine-weather evidence
    weather_event_weights: dict = None

    def __post_init__(self):
        if self.sensor_fault_weights is None:
            self.sensor_fault_weights = {
                "spike": 0.25,
                "cross_variable": 0.20,
                "implausibility": 0.20,
                "non_persistence": 0.15,
                "historical": 0.10,
                "stuck_sensor": 0.10,
            }

        if self.weather_event_weights is None:
            self.weather_event_weights = {
                "multivariate": 0.30,
                "persistence": 0.25,
                "plausibility": 0.25,
                "historical": 0.20,
            }
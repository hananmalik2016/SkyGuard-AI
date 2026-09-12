from pprint import pprint

from src.contextual.contextual_engine import analyze_context
from src.decision_engine.decision_engine import (
    make_context_aware_decision,
)


# =====================================================
# 1. Current observation from Layer 3
# =====================================================

current_data = {
    "station_id": "AWS_001",
    "timestamp": "2026-09-12T14:03:00",

    "temperature": 41.8,
    "pressure": 1008.9,
    "humidity": 71.2,

    "temperature_lag_1": 40.2,
    "temperature_lag_3": 40.0,
    "temperature_lag_5": 39.8,

    "pressure_lag_1": 1008.4,
    "pressure_lag_3": 1008.3,
    "pressure_lag_5": 1008.2,

    "humidity_lag_1": 70.5,
    "humidity_lag_3": 70.1,
    "humidity_lag_5": 69.8,

    "temperature_diff_1": 1.6,
    "pressure_diff_1": 0.5,
    "humidity_diff_1": 0.7,

    "temperature_rate": 1.6,
    "pressure_rate": 0.5,
    "humidity_rate": 0.7,

    "temperature_rolling_mean_5": 40.1,
    "temperature_rolling_mean_15": 39.9,
    "temperature_rolling_std_5": 0.5,

    "pressure_rolling_mean_5": 1008.4,
    "pressure_rolling_mean_15": 1008.5,
    "pressure_rolling_std_5": 0.2,

    "humidity_rolling_mean_5": 70.2,
    "humidity_rolling_mean_15": 69.9,
    "humidity_rolling_std_5": 0.4,

    "gap_detected": False,
    "is_imputed": False,
    "time_diff": 60,
}


# =====================================================
# 2. Recent observations
#    Normally supplied by your data pipeline
# =====================================================

recent_data = [
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-12T13:59:00",
        "temperature": 39.8,
        "pressure": 1008.2,
        "humidity": 69.8,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-12T14:00:00",
        "temperature": 40.0,
        "pressure": 1008.3,
        "humidity": 70.1,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-12T14:01:00",
        "temperature": 40.1,
        "pressure": 1008.4,
        "humidity": 70.2,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-12T14:02:00",
        "temperature": 40.2,
        "pressure": 1008.4,
        "humidity": 70.5,
    },
]


# =====================================================
# 3. Historical observations
# =====================================================

historical_data = [
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-11T14:00:00",
        "temperature": 35.0,
        "pressure": 1010.0,
        "humidity": 60.0,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-10T14:00:00",
        "temperature": 35.5,
        "pressure": 1009.8,
        "humidity": 61.0,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-09T14:00:00",
        "temperature": 36.0,
        "pressure": 1010.2,
        "humidity": 59.5,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-08T14:00:00",
        "temperature": 35.2,
        "pressure": 1010.1,
        "humidity": 60.5,
    },
]


# =====================================================
# 4. Layer 4 output
# =====================================================

anomaly_result = {
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


# =====================================================
# 5. LAYER 5
# =====================================================

contextual_evidence = analyze_context(
    current_data=current_data,
    recent_data=recent_data,
    historical_data=historical_data,
    anomaly_result=anomaly_result,
)


# =====================================================
# 6. LAYER 7
# =====================================================

decision = make_context_aware_decision(
    anomaly_result=anomaly_result,
    contextual_evidence=contextual_evidence,
)


# =====================================================
# 7. Final result
# =====================================================

print("\n========== LAYER 5 ==========")
pprint(contextual_evidence)

print("\n========== LAYER 7 ==========")
pprint(decision)
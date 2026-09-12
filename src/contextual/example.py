import pandas as pd

from .contextual_engine import analyze_context


# ---------------------------------------------------------
# 1. Current observation coming from Layer 3
# ---------------------------------------------------------

current = {
    "station_id": "AWS_001",
    "timestamp": "2026-09-12T14:03:00",

    "temperature": 41.8,
    "pressure": 1008.9,
    "humidity": 71.2,

    # Lag features
    "temperature_lag_1": 40.2,
    "temperature_lag_3": 39.9,
    "temperature_lag_5": 39.5,

    "pressure_lag_1": 1008.4,
    "pressure_lag_3": 1008.2,
    "pressure_lag_5": 1008.0,

    "humidity_lag_1": 70.5,
    "humidity_lag_3": 69.8,
    "humidity_lag_5": 69.0,

    # Difference features
    "temperature_diff_1": 1.6,
    "temperature_abs_diff": 1.6,

    "pressure_diff_1": 0.5,
    "pressure_abs_diff": 0.5,

    "humidity_diff_1": 0.7,
    "humidity_abs_diff": 0.7,

    # Rate features
    # Assuming Layer 3 stores these as units per minute
    "temperature_rate": 1.6,
    "pressure_rate": 0.5,
    "humidity_rate": 0.7,

    # Rolling features
    "temperature_rolling_mean_5": 40.1,
    "temperature_rolling_mean_15": 39.8,
    "temperature_rolling_std_5": 0.5,
    "temperature_rolling_min_15": 38.9,
    "temperature_rolling_max_15": 40.5,

    "pressure_rolling_mean_5": 1008.5,
    "pressure_rolling_mean_15": 1008.3,
    "pressure_rolling_std_5": 0.3,
    "pressure_rolling_min_15": 1007.8,
    "pressure_rolling_max_15": 1008.9,

    "humidity_rolling_mean_5": 70.4,
    "humidity_rolling_mean_15": 70.0,
    "humidity_rolling_std_5": 0.8,
    "humidity_rolling_min_15": 68.5,
    "humidity_rolling_max_15": 72.0,

    # Time features
    "hour_sin": 0.5,
    "hour_cos": -0.5,

    # Data-quality features
    "is_imputed": False,
    "gap_detected": False,
    "time_diff": 60,

    # Layer 3 anomaly fields
    "is_anomaly": True,
    "anomaly_type": "point_anomaly",
}


# ---------------------------------------------------------
# 2. Recent observations
# ---------------------------------------------------------

recent = pd.DataFrame([
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-12T13:59:00",
        "temperature": 39.8,
        "pressure": 1008.3,
        "humidity": 69.5,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-12T14:00:00",
        "temperature": 40.0,
        "pressure": 1008.4,
        "humidity": 69.8,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-12T14:01:00",
        "temperature": 40.1,
        "pressure": 1008.5,
        "humidity": 70.0,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-12T14:02:00",
        "temperature": 40.2,
        "pressure": 1008.4,
        "humidity": 70.5,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-12T14:03:00",
        "temperature": 41.8,
        "pressure": 1008.9,
        "humidity": 71.2,
    },
])


# ---------------------------------------------------------
# 3. Historical observations
# ---------------------------------------------------------

historical = pd.DataFrame([
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-10T14:00:00",
        "temperature": 38.5,
        "pressure": 1009.0,
        "humidity": 68.0,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-10T14:01:00",
        "temperature": 38.7,
        "pressure": 1008.9,
        "humidity": 68.2,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-11T14:00:00",
        "temperature": 39.0,
        "pressure": 1008.7,
        "humidity": 69.0,
    },
    {
        "station_id": "AWS_001",
        "timestamp": "2026-09-11T14:01:00",
        "temperature": 39.2,
        "pressure": 1008.6,
        "humidity": 69.3,
    },
])


# ---------------------------------------------------------
# 4. Layer 4 anomaly result
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 5. Run Layer 5
# ---------------------------------------------------------

result = analyze_context(
    current_data=current,
    recent_data=recent,
    historical_data=historical,
    anomaly_result=anomaly_result,
)


# ---------------------------------------------------------
# 6. Print the result
# ---------------------------------------------------------

import pprint

pprint.pp(result)
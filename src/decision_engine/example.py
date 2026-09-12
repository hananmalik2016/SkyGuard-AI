from pprint import pprint

from .decision_engine import make_context_aware_decision


anomaly_result = {
    "anomaly_score": 0.94,
    "anomaly_status": "anomalous",
}


contextual_evidence = {
    "temporal_context": {
        "temporal_pattern": "sudden_change",
    },

    "rate_persistence": {
        "temperature": {
            "rapid_change": True,
            "persistence": False,
            "stuck_value_indicator": False,
        },

        "pressure": {
            "rapid_change": False,
            "persistence": False,
            "stuck_value_indicator": False,
        },

        "humidity": {
            "rapid_change": False,
            "persistence": False,
            "stuck_value_indicator": False,
        },
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
    anomaly_result,
    contextual_evidence,
)

pprint(result)
def _clamp(value):
    return max(0.0, min(1.0, float(value)))


def _anomaly_strength(anomaly_result):
    score = anomaly_result.get("anomaly_score", 0.0)

    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    return _clamp(score)


def calculate_sensor_fault_evidence(
    anomaly_result,
    contextual_evidence,
    config,
):
    temporal = contextual_evidence.get(
        "temporal_context", {}
    )

    rate_persistence = contextual_evidence.get(
        "rate_persistence", {}
    )

    cross_variable = contextual_evidence.get(
        "cross_variable_context", {}
    )

    meteorological = contextual_evidence.get(
        "meteorological_context", {}
    )

    historical = contextual_evidence.get(
        "historical_context", {}
    )

    # --------------------------------------------------
    # 1. Spike evidence
    # --------------------------------------------------

    sudden_change = (
        temporal.get("temporal_pattern")
        == "sudden_change"
    )

    rapid_change = any(
        isinstance(data, dict)
        and data.get("rapid_change") is True
        for data in rate_persistence.values()
    )

    spike_evidence = 1.0 if (
        sudden_change or rapid_change
    ) else 0.0

    # --------------------------------------------------
    # 2. Cross-variable inconsistency
    # --------------------------------------------------

    consistency = cross_variable.get(
        "overall_consistency"
    )

    cross_variable_evidence = {
        "stable": 1.0,
        "low": 0.8,
        "medium": 0.4,
        "high": 0.0,
    }.get(consistency, 0.0)

    # --------------------------------------------------
    # 3. Meteorological implausibility
    # --------------------------------------------------

    plausibility = meteorological.get(
        "plausibility"
    )

    implausibility_evidence = {
        "low": 1.0,
        "medium": 0.5,
        "high": 0.0,
    }.get(plausibility, 0.0)

    # --------------------------------------------------
    # 4. Non-persistence
    # --------------------------------------------------

    persistent_count = meteorological.get(
        "persistent_change_count"
    )

    if persistent_count is None:
        persistent_count = sum(
            1
            for data in rate_persistence.values()
            if isinstance(data, dict)
            and data.get("persistence") is True
        )

    non_persistence_evidence = (
        1.0 if persistent_count == 0 else 0.0
    )

    # --------------------------------------------------
    # 5. Historical evidence
    # --------------------------------------------------

    historical_evidence = 0.0

    if historical.get("available") is True:

        for variable_data in historical.values():

            if not isinstance(variable_data, dict):
                continue

            consistency_value = variable_data.get(
                "consistency_with_baseline"
            )

            if consistency_value == "low":
                historical_evidence = max(
                    historical_evidence,
                    1.0
                )

            elif consistency_value == "medium":
                historical_evidence = max(
                    historical_evidence,
                    0.5
                )

    # --------------------------------------------------
    # 6. Stuck sensor
    # --------------------------------------------------

    stuck_sensor = any(
        isinstance(data, dict)
        and data.get("stuck_value_indicator") is True
        for data in rate_persistence.values()
    )

    stuck_evidence = (
        1.0 if stuck_sensor else 0.0
    )

    # --------------------------------------------------
    # Weighted combination
    # --------------------------------------------------

    weights = config.sensor_fault_weights

    score = (
        weights["spike"] * spike_evidence
        + weights["cross_variable"] * cross_variable_evidence
        + weights["implausibility"] * implausibility_evidence
        + weights["non_persistence"] * non_persistence_evidence
        + weights["historical"] * historical_evidence
        + weights["stuck_sensor"] * stuck_evidence
    )

    return {
        "score": _clamp(score),
        "components": {
            "spike": spike_evidence,
            "cross_variable": cross_variable_evidence,
            "implausibility": implausibility_evidence,
            "non_persistence": non_persistence_evidence,
            "historical": historical_evidence,
            "stuck_sensor": stuck_evidence,
        },
    }


def calculate_weather_event_evidence(
    contextual_evidence,
    config,
):
    rate_persistence = contextual_evidence.get(
        "rate_persistence", {}
    )

    cross_variable = contextual_evidence.get(
        "cross_variable_context", {}
    )

    meteorological = contextual_evidence.get(
        "meteorological_context", {}
    )

    historical = contextual_evidence.get(
        "historical_context", {}
    )

    # --------------------------------------------------
    # 1. Multiple variables changed
    # --------------------------------------------------

    changed_count = cross_variable.get(
        "number_of_changed_variables",
        0
    )

    if changed_count >= 3:
        multivariate_evidence = 1.0
    elif changed_count == 2:
        multivariate_evidence = 0.7
    else:
        multivariate_evidence = 0.0

    # --------------------------------------------------
    # 2. Persistence
    # --------------------------------------------------

    persistent_count = meteorological.get(
        "persistent_change_count"
    )

    if persistent_count is None:
        persistent_count = sum(
            1
            for data in rate_persistence.values()
            if isinstance(data, dict)
            and data.get("persistence") is True
        )

    persistence_evidence = (
        1.0 if persistent_count >= 1 else 0.0
    )

    # --------------------------------------------------
    # 3. Meteorological plausibility
    # --------------------------------------------------

    plausibility = meteorological.get(
        "plausibility"
    )

    plausibility_evidence = {
        "high": 1.0,
        "medium": 0.5,
        "low": 0.0,
    }.get(plausibility, 0.0)

    # --------------------------------------------------
    # 4. Historical consistency
    # --------------------------------------------------

    historical_evidence = 0.0

    if historical.get("available") is True:

        consistency_values = []

        for variable_data in historical.values():

            if not isinstance(variable_data, dict):
                continue

            value = variable_data.get(
                "consistency_with_baseline"
            )

            if value:
                consistency_values.append(value)

        if consistency_values:

            high_count = consistency_values.count("high")
            medium_count = consistency_values.count("medium")

            if high_count >= 2:
                historical_evidence = 1.0
            elif high_count >= 1 or medium_count >= 1:
                historical_evidence = 0.5

    # --------------------------------------------------
    # Weighted combination
    # --------------------------------------------------

    weights = config.weather_event_weights

    score = (
        weights["multivariate"] * multivariate_evidence
        + weights["persistence"] * persistence_evidence
        + weights["plausibility"] * plausibility_evidence
        + weights["historical"] * historical_evidence
    )

    return {
        "score": _clamp(score),
        "components": {
            "multivariate": multivariate_evidence,
            "persistence": persistence_evidence,
            "plausibility": plausibility_evidence,
            "historical": historical_evidence,
        },
    }


def calculate_communication_evidence(
    contextual_evidence
):
    """
    Only use explicit communication/data-integrity
    evidence.

    We do NOT infer a communication failure simply
    because another contextual feature is missing.
    """

    data_quality = contextual_evidence.get(
        "data_quality_context",
        {}
    )

    explicit_flags = [
        "communication_issue",
        "data_integrity_issue",
        "missing_observation",
        "timestamp_irregularity",
        "duplicate_observation",
        "malformed_packet",
    ]

    detected = any(
        data_quality.get(flag) is True
        for flag in explicit_flags
    )

    return {
        "score": 1.0 if detected else 0.0,
        "detected": detected,
    }


def calculate_all_evidence(
    anomaly_result,
    contextual_evidence,
    config,
):
    sensor_fault = calculate_sensor_fault_evidence(
        anomaly_result,
        contextual_evidence,
        config,
    )

    weather_event = calculate_weather_event_evidence(
        contextual_evidence,
        config,
    )

    communication = calculate_communication_evidence(
        contextual_evidence
    )

    return {
        "sensor_fault": sensor_fault,
        "genuine_weather": weather_event,
        "communication": communication,
        "anomaly_strength": _anomaly_strength(
            anomaly_result
        ),
    }
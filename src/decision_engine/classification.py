def classify_decision(
    anomaly_strength,
    evidence,
    config,
):
    # --------------------------------------------------
    # Step 1: Is this actually anomalous?
    # --------------------------------------------------

    if anomaly_strength < config.anomaly_threshold:
        return {
            "classification": "NORMAL",
            "decision_score": 1.0 - anomaly_strength,
            "decision_reasons": [
                "Anomaly evidence is below the decision threshold."
            ],
        }

    # --------------------------------------------------
    # Step 2: Communication/data issue
    # --------------------------------------------------

    communication = evidence["communication"]

    if communication["detected"]:
        return {
            "classification": "DATA_COMMUNICATION_ISSUE",
            "decision_score": communication["score"],
            "decision_reasons": [
                "Explicit data communication or integrity evidence detected."
            ],
        }

    # --------------------------------------------------
    # Step 3: Calculate candidate scores
    # --------------------------------------------------

    sensor_score = (
        0.35 * anomaly_strength
        + 0.65 * evidence["sensor_fault"]["score"]
    )

    weather_score = (
        0.35 * anomaly_strength
        + 0.65 * evidence["genuine_weather"]["score"]
    )

    # --------------------------------------------------
    # Step 4: Determine whether we have enough evidence
    # --------------------------------------------------

    sensor_components = evidence[
        "sensor_fault"
    ]["components"]

    weather_components = evidence[
        "genuine_weather"
    ]["components"]

    available_components = sum(
        value > 0
        for value in sensor_components.values()
    )

    available_components += sum(
        value > 0
        for value in weather_components.values()
    )

    if available_components == 0:
        return {
            "classification": "UNCERTAIN",
            "decision_score": anomaly_strength,
            "decision_reasons": [
                "High anomaly detected but insufficient contextual evidence."
            ],
        }

    # --------------------------------------------------
    # Step 5: Check conflict
    # --------------------------------------------------

    difference = abs(
        sensor_score - weather_score
    )

    if (
        sensor_score >= config.decision_threshold
        and weather_score >= config.decision_threshold
        and difference < config.conflict_margin
    ):
        return {
            "classification": "UNCERTAIN",
            "decision_score": max(
                sensor_score,
                weather_score
            ),
            "decision_reasons": [
                "Strong sensor-fault and genuine-event evidence conflict."
            ],
        }

    # --------------------------------------------------
    # Step 6: Sensor fault
    # --------------------------------------------------

    if sensor_score >= config.decision_threshold:
        return {
            "classification": "SENSOR_FAULT",
            "decision_score": sensor_score,
            "decision_reasons": [],
        }

    # --------------------------------------------------
    # Step 7: Genuine weather event
    # --------------------------------------------------

    if weather_score >= config.decision_threshold:
        return {
            "classification":
                "GENUINE_METEOROLOGICAL_EVENT",
            "decision_score": weather_score,
            "decision_reasons": [],
        }

    # --------------------------------------------------
    # Step 8: Insufficient evidence
    # --------------------------------------------------

    return {
        "classification": "UNCERTAIN",
        "decision_score": max(
            sensor_score,
            weather_score
        ),
        "decision_reasons": [
            "Available contextual evidence is not strong enough for a confident classification."
        ],
    }
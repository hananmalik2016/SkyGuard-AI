from .schemas import DecisionConfig
from .evidence_scoring import calculate_all_evidence
from .classification import classify_decision


def make_context_aware_decision(
    anomaly_result,
    contextual_evidence,
    config=None,
):
    """
    Combine anomaly detection output and contextual
    evidence into a final classification.
    """

    config = config or DecisionConfig()

    evidence = calculate_all_evidence(
        anomaly_result,
        contextual_evidence,
        config,
    )

    decision = classify_decision(
        evidence["anomaly_strength"],
        evidence,
        config,
    )

    # --------------------------------------------------
    # Generate traceable reasons
    # --------------------------------------------------

    if decision["classification"] == "SENSOR_FAULT":

        components = evidence[
            "sensor_fault"
        ]["components"]

        reasons = []

        if components["spike"] >= 0.6:
            reasons.append(
                "Sudden or rapid sensor change detected."
            )

        if components["cross_variable"] >= 0.6:
            reasons.append(
                "Low cross-variable consistency."
            )

        if components["implausibility"] >= 0.6:
            reasons.append(
                "Low meteorological plausibility."
            )

        if components["non_persistence"] >= 0.6:
            reasons.append(
                "Abnormal change was not persistent."
            )

        if components["historical"] >= 0.6:
            reasons.append(
                "Observation is unusual relative to historical behavior."
            )

        if components["stuck_sensor"] >= 0.6:
            reasons.append(
                "Sensor appears to be producing repeated identical values."
            )

        decision["decision_reasons"] = reasons

    elif (
        decision["classification"]
        == "GENUINE_METEOROLOGICAL_EVENT"
    ):

        components = evidence[
            "genuine_weather"
        ]["components"]

        reasons = []

        if components["multivariate"] >= 0.6:
            reasons.append(
                "Multiple weather variables changed together."
            )

        if components["persistence"] >= 0.6:
            reasons.append(
                "The change persisted across observations."
            )

        if components["plausibility"] >= 0.6:
            reasons.append(
                "The observed pattern is meteorologically plausible."
            )

        if components["historical"] >= 0.6:
            reasons.append(
                "The pattern is consistent with historical behavior."
            )

        decision["decision_reasons"] = reasons

    return {
        "classification":
            decision["classification"],

        "decision_score":
            round(
                decision["decision_score"],
                4
            ),

        "decision_reasons":
            decision["decision_reasons"],

        "evidence": evidence,
    }
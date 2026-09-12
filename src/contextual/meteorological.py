def evaluate_meteorological_plausibility(
    temporal_context: dict,
    rate_persistence: dict,
    cross_variable_context: dict,
    current_data: dict,
) -> dict:

    changed_variables = cross_variable_context.get(
        "number_of_changed_variables",
        0
    )

    rapid_changes = sum(
        1
        for variable_data in rate_persistence.values()
        if isinstance(variable_data, dict)
        and variable_data.get("rapid_change") is True
    )

    persistent_changes = sum(
        1
        for variable_data in rate_persistence.values()
        if isinstance(variable_data, dict)
        and variable_data.get("persistence") is True
    )

    gap_detected = bool(
        current_data.get("gap_detected", False)
    )

    is_imputed = bool(
        current_data.get("is_imputed", False)
    )

    # Insufficient data / degraded data quality.
    if gap_detected and changed_variables == 0:
        plausibility = "insufficient_data"

    # A single rapidly changing variable while the others remain
    # stable is weak contextual support for a meteorological event.
    elif rapid_changes == 1 and changed_variables == 1:
        plausibility = "low"

    # Multiple variables changing and persisting provides stronger
    # contextual support.
    elif (
        changed_variables >= 2
        and persistent_changes >= 1
    ):
        plausibility = "high"

    elif changed_variables >= 2:
        plausibility = "medium"

    elif rapid_changes >= 1:
        plausibility = "low"

    else:
        plausibility = "medium"

    return {
        "plausibility": plausibility,
        "rapid_change": rapid_changes > 0,
        "rapid_change_count": rapid_changes,
        "persistent_change_count": persistent_changes,
        "changed_variable_count": changed_variables,
        "data_quality_concern": (
            gap_detected or is_imputed
        ),
    }
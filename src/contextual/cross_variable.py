import numpy as np
import pandas as pd

from .schemas import CORE_VARIABLES, ContextConfig


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _change_from_current(current_data, variable):
    """
    Prefer Layer 3's diff_1 feature.
    """

    diff = _safe_float(
        current_data.get(f"{variable}_diff_1")
    )

    if diff is not None:
        return diff

    current = _safe_float(
        current_data.get(variable)
    )

    lag = _safe_float(
        current_data.get(f"{variable}_lag_1")
    )

    if current is None or lag is None:
        return None

    return current - lag


def _recent_correlations(recent_data):
    """
    Calculate recent pairwise correlations.

    Correlation is treated as supporting evidence only.
    """

    result = {}

    if recent_data is None or recent_data.empty:
        return result

    available = [
        variable
        for variable in CORE_VARIABLES
        if variable in recent_data.columns
    ]

    if len(available) < 2:
        return result

    numeric = recent_data[available].apply(
        pd.to_numeric,
        errors="coerce"
    )

    correlation = numeric.corr()

    for i, variable_a in enumerate(available):
        for variable_b in available[i + 1:]:
            value = correlation.loc[
                variable_a,
                variable_b
            ]

            if pd.isna(value):
                continue

            result[
                f"{variable_a}_{variable_b}"
            ] = float(value)

    return result


def calculate_cross_variable_context(
    current_data: dict,
    recent_data: pd.DataFrame,
    config: ContextConfig | None = None,
) -> dict:

    config = config or ContextConfig()

    changes = {
        variable: _change_from_current(
            current_data,
            variable
        )
        for variable in CORE_VARIABLES
    }

    significant = {}

    for variable in CORE_VARIABLES:
        change = changes[variable]

        significant[variable] = (
            change is not None
            and abs(change)
            >= config.significant_change_thresholds[variable]
        )

    changed_variables = [
        variable
        for variable in CORE_VARIABLES
        if significant[variable]
    ]

    number_changed = len(changed_variables)

    if number_changed == 0:
        consistency = "stable"
    elif number_changed == 1:
        consistency = "low"
    elif number_changed == 2:
        consistency = "medium"
    else:
        consistency = "high"

    correlations = _recent_correlations(
        recent_data
    )

    return {
        "temperature_change": changes["temperature"],
        "pressure_change": changes["pressure"],
        "humidity_change": changes["humidity"],

        "significant_changes": significant,

        "changed_variables": changed_variables,

        "number_of_changed_variables": number_changed,

        "recent_correlations": correlations,

        "overall_consistency": consistency,
    }
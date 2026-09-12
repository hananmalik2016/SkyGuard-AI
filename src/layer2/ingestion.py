import pandas as pd

from layer1.Validation import validate_dataframe
from preprocessing.preprocessing import preprocess_data


def fetch_and_process_layer2_feed(source_path_or_api: str) -> dict:
    """
    Load weather data, validate it through Layer 1,
    and preprocess the valid records through Layer 2.
    """

    raw_df = pd.read_csv(source_path_or_api)

    validation_result = validate_dataframe(raw_df)

    validated_df = validation_result["valid_df"]

    cleaned_df = preprocess_data(validated_df)

    return {
        "clean_data": cleaned_df,
        "quarantined_errors": validation_result["error_logs"],
        "metrics": {
            "total_ingested": validation_result["total_processed"],
            "passed_validation": validation_result["valid_count"],
            "failed_validation": validation_result["error_count"]
        }
    }
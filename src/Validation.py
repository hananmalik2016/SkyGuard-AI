import pandas as pd
import math
from Schema import AWSObservation
from pydantic import ValidationError


def validate_record(record_dict):
    """
    Validates a single weather observation dictionary against the schema.
    """
# Drop empty cells entirely so Pydantic flags them as explicitly missing
    clean_record = {
        key: val for key, val in record_dict.items() 
        if not (isinstance(val, float) and math.isnan(val))
    }

    try:
        # Check against AWSObservation schema
        validated_data = AWSObservation(**clean_record)
        return {
            "record": validated_data.model_dump(),
            "valid": True,
            "validation_errors": [],
            "validation_warnings": []
        }
    except ValidationError as error:
        # Extract specific error messages for every failed field
        error_messages = [
            f"{err['loc'][0]}: {err['msg']}" for err in error.errors()
        ]
        return {
            "record": clean_record,
            "valid": False,
            "validation_errors": error_messages,
            "validation_warnings": []
        }


def validate_file(file_path):
    """
    Loads a CSV file, identifies duplicates, and validates every row.
    """
    # 1. Load observations
    df = pd.read_csv(file_path)

    # 2. Check for duplicate readings (same station and timestamp)
    duplicate_mask = df.duplicated(subset=["station_id", "timestamp"], keep=False)

    # Convert DataFrame to a list of row dictionaries
    rows = df.to_dict(orient="records")
    validated_results = []

    for index, row in enumerate(rows):
        result = validate_record(row)

        # Add a warning if this row's station_id and timestamp appear multiple times
        if duplicate_mask.iloc[index]:
            result["validation_warnings"].append("duplicate_timestamp")

        validated_results.append(result)

    return validated_results


if __name__ == "__main__":
    # Test the validation pipeline on sample data
    results = validate_file("sample_weather.csv")

    for i, res in enumerate(results, start=1):
        print(f"--- Observation {i} ---")
        print(f"Valid: {res['valid']}")
        if res["validation_errors"]:
            print(f"Errors: {res['validation_errors']}")
        if res["validation_warnings"]:
            print(f"Warnings: {res['validation_warnings']}")
        print(f"Record Data: {res['record']}\n")
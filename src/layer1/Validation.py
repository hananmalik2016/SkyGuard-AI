import pandas as pd
from pydantic import ValidationError
from layer1.Schema import AWSObservation

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Processes an entire raw DataFrame, filtering out invalid rows based on 
    IMD Pydantic schemas and returning a clean DataFrame for Layer 2.
    """
    valid_records = []
    error_logs = []
    
    # Drop completely empty cells first
    cleaned_df = df.dropna(how="all")
    
    for index, row in cleaned_df.iterrows():
        row_dict = row.to_dict()
        try:
            # Enforce Pydantic schema rules
            validated_obs = AWSObservation(**row_dict)
            valid_records.append(validated_obs.model_dump())
        except ValidationError as e:
            error_logs.append({
                "row_index": index,
                "record": row_dict,
                "errors": e.errors()
            })
            
    # Construct the final validated DataFrame for Layer 2 consumption
    valid_df = pd.DataFrame(valid_records) if valid_records else pd.DataFrame(columns=df.columns)
    
    return {
        "valid_df": valid_df,
        "error_logs": error_logs,
        "total_processed": len(df),
        "valid_count": len(valid_df),
        "error_count": len(error_logs)
    }
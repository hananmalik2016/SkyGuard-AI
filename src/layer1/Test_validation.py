import pandas as pd
from layer1.Validation import validate_dataframe
from layer1.Schema import AWSObservation

def test_validate_dataframe_success_and_errors():
    # Create a test dataframe with a mix of valid and invalid records
    raw_data = {
        "station_id": ["AWS_Agra", "AWS_Agra", "AWS_Agra"],
        "timestamp": ["2020-01-01T00:00:00", "2020-01-01T01:00:00", "2020-01-01T02:00:00"],
        "temperature": [25.0, 120.0, None], # 1st is valid, 2nd is out of bounds, 3rd has missing temp
        "pressure": [1015.0, 1010.0, 1005.0],
        "humidity": [50.0, 45.0, 60.0]
    }
    df = pd.DataFrame(raw_data)
    
    result = validate_dataframe(df)
    
    # Assertions
    assert result["total_processed"] == 3
    assert result["valid_count"] == 1
    assert result["error_count"] == 2
    assert isinstance(result["valid_df"], pd.DataFrame)
    assert len(result["error_logs"]) == 2
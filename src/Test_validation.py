import os
from Validation import validate_record, validate_file

def test_perfect_weather_record():
    good_record = {
        "station_id": "AWS_Agra",
        "timestamp": "2020-01-01T00:00:00",
        "temperature": 25.0,
        "pressure": 1015.0,
        "humidity": 50.0
    }
    result = validate_record(good_record)
    assert result["valid"] == True
    assert len(result["validation_errors"]) == 0

def test_impossible_pressure_is_blocked():
    bad_pressure_record = {
        "station_id": "AWS_Agra",
        "timestamp": "2020-01-01T00:00:00",
        "temperature": 25.0,
        "pressure": 1200.0, 
        "humidity": 50.0
    }
    result = validate_record(bad_pressure_record)
    assert result["valid"] == False
    assert any("pressure" in error for error in result["validation_errors"])

def test_missing_temperature_is_caught():
    missing_temp_record = {
        "station_id": "AWS_Agra",
        "timestamp": "2020-01-01T00:00:00",
        "temperature": None, 
        "pressure": 1015.0,
        "humidity": 50.0
    }
    result = validate_record(missing_temp_record)
    assert result["valid"] == False
    assert any("temperature" in error for error in result["validation_errors"])

def test_duplicate_records_generate_warning():
    # 1. Create a temporary CSV with duplicate rows
    test_data = """station_id,timestamp,temperature,pressure,humidity
AWS_01,2020-01-01T00:00:00,25.0,1015.0,50.0
AWS_01,2020-01-01T00:00:00,25.0,1015.0,50.0"""
    
    with open("temp_test.csv", "w") as f:
        f.write(test_data)
        
    # 2. Run the file through your pipeline
    results = validate_file("temp_test.csv")
    
    # 3. Demand that both rows are valid, but both have the duplicate warning
    assert results[0]["valid"] == True
    assert "duplicate_timestamp" in results[0]["validation_warnings"]
    assert results[1]["valid"] == True
    assert "duplicate_timestamp" in results[1]["validation_warnings"]
    
    # Clean up the temporary file
    os.remove("temp_test.csv")
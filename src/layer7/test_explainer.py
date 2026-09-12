import pytest
import pandas as pd
from sklearn.ensemble import IsolationForest
from explainer import explain_weather_anomaly

def test_explain_weather_anomaly_success():
    # 1. Setup a dummy model for testing
    normal_data = pd.DataFrame({
        "temperature": [25.0, 26.1, 24.5],
        "pressure": [1015.0, 1012.0, 1014.0],
        "humidity": [50.0, 55.0, 48.0]
    })
    model = IsolationForest(random_state=42).fit(normal_data)
    
    # 2. Setup a 1-row anomaly (Extreme Temperature)
    anomalous_record = pd.DataFrame({
        "temperature": [45.0], 
        "pressure": [1010.0], 
        "humidity": [45.0]
    })
    
    # 3. Execute Layer 7
    result = explain_weather_anomaly(model, anomalous_record)
    
    # 4. Assertions
    assert result["status"] == "success"
    assert "primary_driver" in result
    assert isinstance(result["feature_impacts"], dict)

def test_explain_weather_anomaly_multi_row_error():
    # Test that it correctly rejects multi-row dataframes
    model = IsolationForest(random_state=42)
    bad_record = pd.DataFrame({"temp": [45.0, 46.0], "pres": [1010.0, 1011.0]})
    
    result = explain_weather_anomaly(model, bad_record)
    assert result["status"] == "error"
    assert "exactly one anomalous record" in result["message"]
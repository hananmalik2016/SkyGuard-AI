import sys
import os

# Force Python to look inside the 'src' directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from layer2.ingestion import fetch_and_process_layer2_feed

# 1. Create a dummy weather dataset CSV to simulate Layer 2 input feed
sample_data = {
    "station_id": ["AWS_Agra", "AWS_Agra", "AWS_Agra"],
    "timestamp": ["2020-01-01T00:00:00", "2020-01-01T01:00:00", "2020-01-01T02:00:00"],
    "temperature": [28.5, 150.0, 22.1],
    "pressure": [1012.5, 1008.0, 1010.0],
    "humidity": [65.0, 55.0, 70.0]
}
test_csv_path = "sample_weather_feed.csv"
pd.DataFrame(sample_data).to_csv(test_csv_path, index=False)

try:
    # 2. Execute the ingestion and validation pipeline
    result = fetch_and_process_layer2_feed(test_csv_path)
    
    # 3. Print the live operational metrics
    print(" Pipeline Execution Successful!")
    print(f"Total Records Ingested: {result['metrics']['total_ingested']}")
    print(f"Passed Validation (Layer 2 Clean Data): {result['metrics']['passed_validation']}")
    print(f"Failed Validation (Quarantined Errors): {result['metrics']['failed_validation']}")
    
    print("\n Clean Data Preview:")
    print(result["clean_data"])
    
finally:
    # Cleanup dummy file
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
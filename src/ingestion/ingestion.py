import pandas as pd


REQUIRED_COLUMNS = [
    "station_id",
    "timestamp",
    "temperature",
    "pressure",
    "humidity"
]


def ingest_data(file_path):
    """Load raw weather data from a CSV file."""

    df = pd.read_csv(file_path)

    return df
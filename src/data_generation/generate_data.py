import numpy as np
import pandas as pd


def generate_weather_data(
    num_stations=5,
    observations_per_station=1000,
    seed=42
):
    np.random.seed(seed)

    records = []

    start_time = pd.Timestamp("2026-01-01 00:00:00")

    for station in range(num_stations):

        station_id = f"AWS_{station + 1:03d}"

        timestamps = pd.date_range(
            start=start_time,
            periods=observations_per_station,
            freq="10min"
        )

        hours = np.arange(observations_per_station) / 6

        temperature = (
            25
            + 5 * np.sin(2 * np.pi * hours / 24)
            + np.random.normal(0, 0.8, observations_per_station)
        )

        pressure = (
            1013
            + np.random.normal(0, 2, observations_per_station)
        )

        humidity = (
            65
            - 10 * np.sin(2 * np.pi * hours / 24)
            + np.random.normal(0, 3, observations_per_station)
        )

        station_data = pd.DataFrame({
            "station_id": station_id,
            "timestamp": timestamps,
            "temperature": temperature,
            "pressure": pressure,
            "humidity": humidity
        })

        records.append(station_data)

    return pd.concat(records, ignore_index=True)


if __name__ == "__main__":
    df = generate_weather_data()

    df.to_csv(
        "data/raw/weather_data.csv",
        index=False
    )

    print(f"Generated {len(df)} observations")
    print(df.head())
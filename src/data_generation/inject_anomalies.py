import pandas as pd


def inject_anomalies(df):
    """Inject controlled anomalies into synthetic weather data."""

    result = df.copy()

    result["is_anomaly"] = False
    result["anomaly_type"] = "normal"

    # --------------------------------------------------
    # 1. Temperature spike
    # --------------------------------------------------
    idx = result[
        result["station_id"].eq("AWS_001")
    ].index[100]

    result.loc[idx, "temperature"] = 75
    result.loc[idx, "is_anomaly"] = True
    result.loc[idx, "anomaly_type"] = "temperature_spike"

    # --------------------------------------------------
    # 2. Pressure drop
    # --------------------------------------------------
    idx = result[
        result["station_id"].eq("AWS_002")
    ].index[200]

    result.loc[idx, "pressure"] = 950
    result.loc[idx, "is_anomaly"] = True
    result.loc[idx, "anomaly_type"] = "pressure_drop"

    # --------------------------------------------------
    # 3. Humidity anomaly
    # --------------------------------------------------
    idx = result[
        result["station_id"].eq("AWS_003")
    ].index[300]

    result.loc[idx, "humidity"] = 5
    result.loc[idx, "is_anomaly"] = True
    result.loc[idx, "anomaly_type"] = "humidity_anomaly"

    # --------------------------------------------------
    # 4. Sudden temperature change
    # --------------------------------------------------
    station_4 = result[
        result["station_id"].eq("AWS_004")
    ].index

    for i, value in enumerate([20, 21, 45, 46]):
        idx = station_4[400 + i]
        result.loc[idx, "temperature"] = value
        result.loc[idx, "is_anomaly"] = True
        result.loc[idx, "anomaly_type"] = "sudden_temperature_change"

    # --------------------------------------------------
    # 5. Persistent sensor fault
    # --------------------------------------------------
    station_5 = result[
        result["station_id"].eq("AWS_005")
    ].index

    for i in range(500, 510):
        idx = station_5[i]
        result.loc[idx, "temperature"] = 30
        result.loc[idx, "is_anomaly"] = True
        result.loc[idx, "anomaly_type"] = "persistent_sensor_fault"

    return result


if __name__ == "__main__":
    df = pd.read_csv("data/raw/weather_data.csv")

    df = inject_anomalies(df)

    df.to_csv(
        "data/raw/weather_data_with_anomalies.csv",
        index=False
    )

    print("Anomalies injected successfully")
    print(df["anomaly_type"].value_counts())
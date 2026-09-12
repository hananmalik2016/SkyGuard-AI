import pandas as pd

from src.features import create_all_features


def test_layer3_pipeline():
    df = pd.DataFrame({
        "station_id": ["A"] * 6 + ["B"] * 6,
        "timestamp": list(pd.date_range("2026-01-01", periods=6, freq="10min")) * 2,
        "temperature": [20, 21, 22, 21, 23, 24] * 2,
        "pressure": [1000, 1001, 1002, 1001, 1003, 1004] * 2,
        "humidity": [60, 61, 62, 61, 63, 64] * 2,
    })

    result = create_all_features(df)

    assert len(result) == 12
    assert "temperature_lag_1" in result.columns
    assert "temperature_lag_3" in result.columns
    assert "temperature_lag_5" in result.columns
    assert "temperature_diff_1" in result.columns
    assert "temperature_rate" in result.columns
    assert "temperature_rolling_mean_5" in result.columns
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns

    station_a = result[result["station_id"] == "A"]

    assert pd.isna(station_a.iloc[0]["temperature_lag_1"])
    assert station_a.iloc[1]["temperature_lag_1"] == 20
    assert station_a.iloc[5]["temperature_lag_5"] == 20

    assert station_a.iloc[1]["temperature_diff_1"] == 1
    assert station_a.iloc[1]["temperature_rate"] == 0.1

    assert station_a.iloc[5]["time_diff"] == 600


if __name__ == "__main__":
    test_layer3_pipeline()
    print("Layer 3 tests passed")

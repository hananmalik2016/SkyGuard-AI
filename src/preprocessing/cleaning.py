import pandas as pd


NUMERIC_COLUMNS = [
    "temperature",
    "pressure",
    "humidity"
]


def clean_data(df):
    """Clean and prepare validated AWS weather data."""

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce"
    )

    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    result["station_id"] = result["station_id"].astype("string")

    result = result.dropna(
        subset=["timestamp"]
    )

    result = result.sort_values(
        ["station_id", "timestamp"]
    ).reset_index(drop=True)

    duplicate_mask = result.duplicated(
        subset=["station_id", "timestamp"],
        keep=False
    )

    result["is_duplicate"] = duplicate_mask

    result = result.drop_duplicates(
        subset=["station_id", "timestamp"],
        keep="first"
    ).reset_index(drop=True)

    result["time_diff"] = (
        result.groupby("station_id")["timestamp"]
        .diff()
    )

    result["gap_detected"] = (
        result["time_diff"] > pd.Timedelta(hours=1)
    )

    result["is_imputed"] = False
    result["imputation_method"] = pd.NA

    for column in NUMERIC_COLUMNS:
        missing_before = result[column].isna()

        result[column] = (
            result.groupby("station_id")[column]
            .transform(
                lambda x: x.interpolate(
                    method="linear",
                    limit=2,
                    limit_area="inside"
                )
            )
        )

        imputed = (
            missing_before
            & result[column].notna()
        )

        result.loc[imputed, "is_imputed"] = True
        result.loc[imputed, "imputation_method"] = "linear"

    return result
import pandas as pd
import shap
import numpy as np

def explain_weather_anomaly(model, anomalous_record: pd.DataFrame) -> dict:
    try:
        if len(anomalous_record) != 1:
            raise ValueError("Explainability engine requires exactly one anomalous record at a time.")

        explainer = shap.Explainer(model)
        shap_values = explainer(anomalous_record)

        feature_names = anomalous_record.columns.tolist()
        impacts = shap_values.values[0]

        impact_map = {
            str(feature): float(abs(impact)) 
            for feature, impact in zip(feature_names, impacts)
        }

        sorted_impacts = dict(sorted(impact_map.items(), key=lambda item: item[1], reverse=True))

        return {
            "status": "success",
            "primary_driver": list(sorted_impacts.keys())[0],
            "feature_impacts": sorted_impacts
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Layer 7 SHAP generation failed: {str(e)}"
        }
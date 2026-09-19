from pathlib import Path
import joblib
import pandas as pd
from src.feature_engineering import prepare_data

# Rutas de artefactos
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"


def calculate_financial_impact(
    monthly_charges: float, churn_prob: float, tenure: int
):
    """Calcula el valor financiero en riesgo y el potencial de retorno de inversión (ROI)."""
    # Estimación de CLV a 12 meses vista
    horizon_months = 12
    clv_in_risk = monthly_charges * horizon_months * churn_prob

    # Supuesto de campaña de retención (30% de efectividad con costo de $30 USD)
    retention_cost = 30.0
    expected_recovery = max(0.0, (clv_in_risk * 0.30) - retention_cost)

    return {
        "clv_en_riesgo": round(clv_in_risk, 2),
        "recuperacion_estimada": round(expected_recovery, 2),
    }


def predict_churn(input_df: pd.DataFrame) -> dict:
    """Carga modelos, ejecuta predicción y retorna probabilidades e impacto económico."""
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    df_prepared = prepare_data(input_df)
    X = df_prepared.drop(columns=["Churn", "customerID"], errors="ignore")

    X_transformed = preprocessor.transform(X)
    prob = model.predict_proba(X_transformed)[:, 1][0]
    pred = int(prob >= 0.5)

    monthly_charges = float(input_df["MonthlyCharges"].iloc[0])
    tenure = int(input_df["tenure"].iloc[0])

    financials = calculate_financial_impact(monthly_charges, prob, tenure)

    return {
        "churn_prediction": pred,
        "churn_probability": round(prob, 4),
        "clv_en_riesgo": financials["clv_en_riesgo"],
        "recuperacion_estimada": financials["recuperacion_estimada"],
    }
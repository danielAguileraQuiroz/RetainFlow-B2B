from pathlib import Path
import joblib
import pandas as pd
from src.feature_engineering import prepare_data
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"

# Cargar modelos en memoria una sola vez al importar el módulo
MODEL = joblib.load(MODEL_PATH)
PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)


def process_batch_file(file_path: str) -> tuple[pd.DataFrame, dict]:
    # Cargar según extensión
    if file_path.endswith(".csv"):
        df_raw = pd.read_csv(file_path)
    else:
        df_raw = pd.read_excel(file_path)

    df_prepared = prepare_data(df_raw.copy())
    X = df_prepared.drop(columns=["Churn", "customerID"], errors="ignore")

    # Usar las instancias ya cargadas en memoria
    X_prep = PREPROCESSOR.transform(X)
    probs = MODEL.predict_proba(X_prep)[:, 1]

    # Copia para resultados
    df_result = df_raw.copy()
    df_result["Probabilidad_Churn"] = (probs * 100).round(2)
    df_result["Riesgo_Categoria"] = pd.cut(
        probs,
        bins=[-0.01, 0.30, 0.60, 1.0],
        labels=["Bajo", "Medio", "Alto"],
    )

    # Cálculo de CLV en riesgo a 12 meses
    df_result["CLV_En_Riesgo_USD"] = (
        df_result["MonthlyCharges"] * 12 * probs
    ).round(2)

    # Ordenar de mayor a menor riesgo financiero
    df_result = df_result.sort_values(
        by="CLV_En_Riesgo_USD", ascending=False
    ).reset_index(drop=True)

    # Métricas agregadas para directivos
    summary = {
        "total_clientes": len(df_result),
        "clientes_alto_riesgo": int((df_result["Riesgo_Categoria"] == "Alto").sum()),
        "pérdida_total_estimada_usd": float(df_result["CLV_En_Riesgo_USD"].sum()),
        "ticket_medio_riesgo_usd": float(
            df_result[df_result["Riesgo_Categoria"] == "Alto"][
                "CLV_En_Riesgo_USD"
            ].mean()
            or 0.0
        )
    }
    
    return df_result, summary
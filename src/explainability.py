from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from src.feature_engineering import prepare_data

# Rutas de artefactos
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"

# Mapeo de nombres para visualización de negocio
FEATURE_TRANSLATION = {
    "tenure": "Antigüedad (meses)",
    "MonthlyCharges": "Cargo Mensual ($)",
    "TotalCharges": "Cargos Totales ($)",
    "LTV_Proxy": "Valor de Vida (LTV)",
    "Contract_Month-to-month": "Contrato: Mes a Mes",
    "Contract_One year": "Contrato: 1 Año",
    "Contract_Two year": "Contrato: 2 Años",
    "InternetService_Fiber optic": "Internet: Fibra Óptica",
    "InternetService_DSL": "Internet: DSL",
    "InternetService_No": "Sin Servicio de Internet",
    "PaymentMethod_Electronic check": "Pago: Cheque Electrónico",
    "PaymentMethod_Mailed check": "Pago: Cheque por Correo",
    "PaymentMethod_Bank transfer (automatic)": "Pago: Transferencia Bancaria",
    "PaymentMethod_Credit card (automatic)": "Pago: Tarjeta de Crédito",
    "OnlineSecurity_No": "Seguridad Online: No",
    "TechSupport_No": "Soporte Técnico: No",
    "PaperlessBilling_Yes": "Facturación Electrónica: Sí",
    "SeniorCitizen": "Tercera Edad",
}


def clean_feature_name(name: str) -> str:
    """Elimina prefijos num__/cat__ de scikit-learn y aplica la traducción de negocio."""
    raw_name = name.split("__")[-1] if "__" in name else name
    return FEATURE_TRANSLATION.get(raw_name, raw_name)


def generate_shap_plot(
    input_data: pd.DataFrame, output_path: str = "shap_waterfall.png"
) -> str:
    """Calcula valores SHAP para una instancia y guarda el gráfico Waterfall con nombres limpios."""
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    df_prepared = prepare_data(input_data.copy())
    X = df_prepared.drop(columns=["Churn", "customerID"], errors="ignore")

    X_prep = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    clean_names = [clean_feature_name(f) for f in feature_names]
    X_prep_df = pd.DataFrame(X_prep, columns=clean_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_prep_df)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    return output_path
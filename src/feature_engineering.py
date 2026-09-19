import numpy as np
import pandas as pd


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformaciones de ingeniería de características al dataset."""
    df_clean = df.copy()

    # Asegurar tipos numéricos
    df_clean["MonthlyCharges"] = pd.to_numeric(
        df_clean["MonthlyCharges"], errors="coerce"
    )
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"], errors="coerce"
    )
    df_clean["tenure"] = pd.to_numeric(df_clean["tenure"], errors="coerce")

    # Imputar valores nulos en TotalCharges
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(
        df_clean["MonthlyCharges"] * df_clean["tenure"]
    )

    # Crear feature de valor estimado de cliente (LTV Proxy)
    df_clean["LTV_Proxy"] = df_clean["MonthlyCharges"] * np.where(
        df_clean["tenure"] == 0, 1, df_clean["tenure"]
    )

    return df_clean
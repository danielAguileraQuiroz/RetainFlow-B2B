import sys
from pathlib import Path

# Configurar el path para asegurar importaciones de src
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tempfile
import gradio as gr
import pandas as pd
from src.batch_processing import process_batch_file
from src.explainability import generate_shap_plot
from src.predict import predict_churn


# --- LÓGICA PESTAÑA 1: DIAGNÓSTICO INDIVIDUAL ---
def predict_individual_interface(
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
):
    input_data = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": 1 if SeniorCitizen == "Sí" else 0,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges,
            }
        ]
    )

    res = predict_churn(input_data)

    estado = (
        "⚠️ ALTO RIESGO DE CHURN"
        if res["churn_prediction"] == 1
        else "✅ CLIENTE RETENIDO / BAJO RIESGO"
    )
    prob_str = f"{res['churn_probability']:.2%}"
    clv_str = f"${res['clv_en_riesgo']:,.2f} USD"

    if res["churn_prediction"] == 1:
        accion = (
            f"💡 Sugerencia: Ofrecer migración a Contrato Anual con 15% de descuento o paquete de Soporte Técnico.\n"
            f"💰 Ahorro/Recuperación estimada tras campaña: ${res['recuperacion_estimada']:,.2f} USD"
        )
    else:
        accion = "✅ Sin acción requerida. Mantener monitoreo pasivo."

    shap_img_path = generate_shap_plot(input_data)

    return estado, prob_str, clv_str, accion, shap_img_path


# --- LÓGICA PESTAÑA 2: PROCESAMIENTO MASIVO BATCH ---
def process_batch_interface(file_obj):
    if file_obj is None:
        return "Por favor, sube un archivo CSV o Excel válido.", None, None

    df_result, summary = process_batch_file(file_obj.name)

    resumen_txt = (
        f"📊 **RESUMEN EJECUTIVO DE LA CARTERA**\n\n"
        f"• Total de clientes analizados: **{summary['total_clientes']:,}**\n"
        f"• Clientes en Alto Riesgo de Fuga: **{summary['clientes_alto_riesgo']:,}**\n"
        f"• Pérdida Total Estimada (CLV en Riesgo): **${summary['pérdida_total_estimada_usd']:,.2f} USD**\n"
        f"• Ticket Medio en Riesgo (Clientes Alto Riesgo): **${summary['ticket_medio_riesgo_usd']:,.2f} USD**"
    )

    # Crear archivo temporal para descarga
    temp_dir = tempfile.gettempdir()
    out_path = Path(temp_dir) / "Reporte_Churn_Priorizado.xlsx"
    df_result.to_excel(out_path, index=False)

    return resumen_txt, df_result.head(15), str(out_path)


# --- CONSTRUCCIÓN DE LA INTERFAZ CON GRADIO BLOCKS ---
with gr.Blocks(title="RetainFlow B2B — Platform Enterprise") as demo:
    gr.Markdown(
        """
        # 🛡️ RetainFlow B2B — Customer Retention & Financial Risk Platform
        ### *Plataforma de Inteligencia Artificial Explicable para la Prevención de Churn y Maximización de ROI*
        """
    )

    with gr.Tabs():
        # PESTAÑA 1
        with gr.TabItem("👤 Diagnóstico Individual & Explicabilidad"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📋 Parámetros del Cliente")
                    gender = gr.Dropdown(
                        ["Male", "Female"], label="Género", value="Female"
                    )
                    SeniorCitizen = gr.Dropdown(
                        ["No", "Sí"], label="Jubilado / Mayor 65", value="No"
                    )
                    Partner = gr.Dropdown(
                        ["Yes", "No"], label="Tiene Pareja", value="No"
                    )
                    Dependents = gr.Dropdown(
                        ["Yes", "No"], label="Tiene Dependientes", value="No"
                    )
                    tenure = gr.Slider(
                        0,
                        72,
                        value=1,
                        step=1,
                        label="Antigüedad en meses (Tenure)",
                    )
                    PhoneService = gr.Dropdown(
                        ["Yes", "No"], label="Servicio Telefónico", value="Yes"
                    )
                    MultipleLines = gr.Dropdown(
                        ["Yes", "No", "No phone service"],
                        label="Líneas Múltiples",
                        value="No",
                    )
                    InternetService = gr.Dropdown(
                        ["DSL", "Fiber optic", "No"],
                        label="Servicio Internet",
                        value="Fiber optic",
                    )
                    OnlineSecurity = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Seguridad Online",
                        value="No",
                    )
                    OnlineBackup = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Copia de Seguridad",
                        value="No",
                    )
                    DeviceProtection = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Protección Dispositivo",
                        value="No",
                    )
                    TechSupport = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Soporte Técnico",
                        value="No",
                    )
                    StreamingTV = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Streaming TV",
                        value="No",
                    )
                    StreamingMovies = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Streaming Películas",
                        value="No",
                    )
                    Contract = gr.Dropdown(
                        ["Month-to-month", "One year", "Two year"],
                        label="Tipo de Contrato",
                        value="Month-to-month",
                    )
                    PaperlessBilling = gr.Dropdown(
                        ["Yes", "No"],
                        label="Facturación Electrónica",
                        value="Yes",
                    )
                    PaymentMethod = gr.Dropdown(
                        [
                            "Electronic check",
                            "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)",
                        ],
                        label="Método de Pago",
                        value="Electronic check",
                    )
                    MonthlyCharges = gr.Number(
                        label="Cargo Mensual ($)", value=70.35
                    )
                    TotalCharges = gr.Number(
                        label="Cargos Totales ($)", value=70.35
                    )

                    btn_individual = gr.Button(
                        "🔍 Analizar Riesgo Financiero", variant="primary"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("#### 📈 Diagnóstico & Impacto Monetario")
                    out_estado = gr.Textbox(
                        label="Diagnóstico del Modelo", interactive=False
                    )
                    out_prob = gr.Textbox(
                        label="Probabilidad de Abandono", interactive=False
                    )
                    out_clv = gr.Textbox(
                        label="💰 Pérdida Estimada (CLV en Riesgo a 12 meses)",
                        interactive=False,
                    )
                    out_accion = gr.Textbox(
                        label="🎯 Next-Best-Action (Recomendación Comercial)",
                        lines=3,
                        interactive=False,
                    )
                    out_shap = gr.Image(
                        label="Factores Determinantes (SHAP Waterfall)"
                    )

            btn_individual.click(
                fn=predict_individual_interface,
                inputs=[
                    gender,
                    SeniorCitizen,
                    Partner,
                    Dependents,
                    tenure,
                    PhoneService,
                    MultipleLines,
                    InternetService,
                    OnlineSecurity,
                    OnlineBackup,
                    DeviceProtection,
                    TechSupport,
                    StreamingTV,
                    StreamingMovies,
                    Contract,
                    PaperlessBilling,
                    PaymentMethod,
                    MonthlyCharges,
                    TotalCharges,
                ],
                outputs=[
                    out_estado,
                    out_prob,
                    out_clv,
                    out_accion,
                    out_shap,
                ],
            )

        # PESTAÑA 2
        with gr.TabItem("📁 Análisis Masivo Cartera (Batch)"):
            gr.Markdown(
                """
                ### Carga de Cartera Masiva de Clientes
                Sube un archivo `.csv` o `.xlsx` con la base de datos de clientes para evaluar el riesgo financiero global y descargar la lista priorizada.
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Subir Archivo (.csv / .xlsx)",
                        file_types=[".csv", ".xlsx"],
                    )
                    btn_batch = gr.Button(
                        " Procesar Cartera Completa", variant="primary"
                    )

                with gr.Column(scale=1):
                    batch_summary = gr.Markdown("Esperando archivo...")

            gr.Markdown("#### 📋 Vista Previa de Clientes Priorizados por Riesgo")
            batch_table = gr.Dataframe(interactive=False)
            batch_download = gr.File(
                label="📥 Descargar Reporte Completo en Excel"
            )

            btn_batch.click(
                fn=process_batch_interface,
                inputs=[file_input],
                outputs=[batch_summary, batch_table, batch_download],
            )

if __name__ == "__main__":
    demo.launch()
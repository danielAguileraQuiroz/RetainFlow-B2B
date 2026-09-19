# 🛡️ RetainFlow B2B — Customer Retention & Financial Risk Platform

> **Plataforma B2B de Inteligencia Artificial Explicable para la Prevención de Churn, Estimación de Pérdida Financiera (CLV) y Automatización de Retención.**

---

## 📌 Visión General

**RetainFlow B2B** es una solución Saas / Enterprise orientada a empresas de modelo de suscripción o pago recurrente. A diferencia de los modelos tradicionales de machine learning que solo entregan una probabilidad abstracta, **RetainFlow B2B** traduce el riesgo técnico en **dinero en riesgo ($ USD)**, identifica la causa raíz mediante **SHAP (Shapley Additive exPlanations)** y recomienda la **Siguiente Mejor Acción (Next-Best-Action)** para maximizar el ROI de las campañas de retención.

---

## 🚀 Características Clave

- **👤 Diagnóstico Individual Granular:**
  - Inferencia en tiempo real del riesgo de abandono sobre modelos XGBoost optimizados.
  - Estimación monetaria del **Customer Lifetime Value (CLV)** en riesgo a 12 meses vista.
  - Cálculo de la **Recuperación Financiera Estimada** tras la aplicación de incentivos de retención.
  - Explicabilidad local mediante gráficos **SHAP Waterfall** traducidos a terminología de negocio.

- **📁 Análisis Masivo por Lotes (Batch Processing):**
  - Carga masiva de datasets de clientes en formato `.csv` o `.xlsx`.
  - Resumen ejecutivo automático para directivos (Total Clientes, Clientes en Alto Riesgo, Pérdida Global Estimada en $ USD y Ticket Medio en Riesgo).
  - Exportación de listados priorizados en Excel ordenados automáticamente de mayor a menor riesgo financiero.

- **🏛️ Arquitectura Enterprise Modular:**
  - Separación limpia de responsabilidades (`src/` para lógica de negocio, ingeniería de características, inferencia y batching; `models/` para artefactos de ML; `app_enterprise.py` para la interfaz).

---

## 🏗️ Arquitectura del Proyecto

```text
RetainFlow B2B/
├── data/                       # Datasets de prueba y plantillas (.csv)
│   └── template_churn_batch.csv
├── models/                     # Artefactos del modelo (XGBoost & Preprocesadores)
│   ├── churn_model.joblib
│   └── preprocessor.joblib
├── src/                        # Lógica modular
│   ├── __init__.py
│   ├── feature_engineering.py  # Preprocesamiento y creación de LTV Proxy
│   ├── predict.py              # Inferencia y cálculo de métricas financieras
│   ├── explainability.py       # Explicabilidad local con SHAP y mapeo de variables
│   └── batch_processing.py     # Procesamiento masivo B2B y métricas ejecutivas
├── app_enterprise.py           # Dashboard interactivo en Gradio
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación oficial
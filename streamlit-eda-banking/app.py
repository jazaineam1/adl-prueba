import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import openai
import os

# Configuración de la API Key de OpenAI
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv('OPENAI_API_KEY')

# Función para generar explicaciones usando GPT-3.5
def generar_explicacion(caracteristicas, valores_shap):
    prompt = (
        "Dado los siguientes valores SHAP y las características correspondientes, explica el impacto en la predicción del modelo. "
        "Describe cómo cada característica contribuye positiva o negativamente, describelo claramente.\n\n"
        f"Características: {caracteristicas}\n"
        f"Valores SHAP: {valores_shap}\n"
        "Explicación:"
    )

    respuesta = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=350
    )

    explicacion = respuesta['choices'][0]['message']['content']
    return explicacion

# Cargar el modelo y el escalador
with open('xgboost_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
explainer = model_data['explainer']
feature_names = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'ingresos', 'egresos', 'saldo_captacion', 'saldo_tdc', 'monto_trx_tdc', 'cantidad_trx_tdc', 'saldo_lb', 'saldo_veh', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 'marital_married', 'marital_single', 'marital_unknown', 'education_secondary', 'education_tertiary', 'education_unknown', 'default_1', 'housing_1', 'contact_telephone', 'contact_unknown', 'poutcome_other', 'poutcome_success', 'poutcome_unknown', 'deposit_1', 'tenencia_ahorros_1.0', 'tenencia_ahorros_unknown', 'tenencia_corriente_1.0', 'tenencia_corriente_unknown', 'tenencia_cdt_1.0', 'tenencia_cdt_unknown', 'tenencia_tdc_1.0', 'tenencia_lb_1.0', 'tenencia_vehiculo_1.0']

# Configuración de Streamlit
st.set_page_config(page_title="Predicción de Crédito Libre inversión", layout="wide")

st.title("Predicción de Crédito con XGBoost de Libre inversión")
st.write("Por favor, ingrese la información del cliente para predecir la probabilidad de aprobación del crédito.")

# Campos de entrada
cols = st.columns(5)
age = cols[0].number_input("Edad", min_value=18, max_value=100, value=30)
balance = cols[1].number_input("Saldo", value=1000)
duration = cols[2].number_input("Duración de la Última Llamada", value=100)
campaign = cols[3].number_input("Número de Contactos", value=1)
pdays = cols[4].number_input("Días Desde el Último Contacto", value=-1)
previous = cols[0].number_input("Contactos Previos", value=0)
ingresos = cols[1].number_input("Ingresos Mensuales", value=20000)
egresos = cols[2].number_input("Egresos Mensuales", value=1500)
saldo_captacion = cols[3].number_input("Saldo en Captación", value=5000)
saldo_tdc = cols[4].number_input("Saldo en Tarjetas de Crédito", value=1000)
monto_trx_tdc = cols[0].number_input("Monto Total de Transacciones con Tarjetas de Crédito", value=500)
cantidad_trx_tdc = cols[1].number_input("Número de Transacciones con Tarjetas de Crédito", value=0)

job = cols[2].selectbox("Trabajo", ['blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
marital = cols[3].selectbox("Estado Civil", ['married', 'single', 'unknown'])
education = cols[4].selectbox("Nivel de Educación", ['secondary', 'tertiary', 'unknown'])
default = cols[0].selectbox("Tiene Créditos en Mora", ['0', '1'])
housing = cols[1].selectbox("Tiene Crédito de Vivienda", ['0', '1'])
contact = cols[2].selectbox("Tipo de Contacto", ['unknown', 'telephone'])
poutcome = cols[3].selectbox("Resultado de la Última Campaña", ['unknown', 'other', 'success'])
deposit = cols[4].selectbox("Tiene Tarjetas Débito", ['0', '1'])
tenencia_ahorros = cols[0].selectbox("Tiene Cuentas de Ahorros", ['0.0', '1.0', 'unknown'])
tenencia_corriente = cols[1].selectbox("Tiene Cuentas Corriente", ['0.0', '1.0', 'unknown'])
tenencia_cdt = cols[2].selectbox("Tiene Productos de CDT", ['0.0', '1.0', 'unknown'])
tenencia_tdc = cols[3].selectbox("Tiene Tarjetas de Crédito", ['0.0', '1.0'])
tenencia_lb = cols[4].selectbox("Tiene Créditos de Libranza", ['0.0', '1.0'])
tenencia_vehiculo = cols[0].selectbox("Tiene Crédito de Vehículos", ['0.0', '1.0'])

# Construir DataFrame para la entrada del cliente
client_data = pd.DataFrame([{
    'age': age, 'balance': balance, 'duration': duration, 'campaign': campaign,
    'pdays': pdays, 'previous': previous, 'ingresos': ingresos, 'egresos': egresos,
    'saldo_captacion': saldo_captacion, 'saldo_tdc': saldo_tdc, 'monto_trx_tdc': monto_trx_tdc,
    'cantidad_trx_tdc': cantidad_trx_tdc, 'saldo_lb': 0, 'saldo_veh': 0,
    'job_' + job: 1, 'marital_' + marital: 1, 'education_' + education: 1,
    'default_' + default: 1, 'housing_' + housing: 1, 'contact_' + contact: 1,
    'poutcome_' + poutcome: 1, 'deposit_' + deposit: 1, 'tenencia_ahorros_' + tenencia_ahorros: 1,
    'tenencia_corriente_' + tenencia_corriente: 1, 'tenencia_cdt_' + tenencia_cdt: 1,
    'tenencia_tdc_' + tenencia_tdc: 1, 'tenencia_lb_' + tenencia_lb: 1, 'tenencia_vehiculo_' + tenencia_vehiculo: 1
}])

# Asegurar que todas las columnas de características estén presentes
for col in feature_names:
    if col not in client_data.columns:
        client_data[col] = 0

# Ordenar el DataFrame según feature_names
client_data = client_data[feature_names]

# Escalar los datos de entrada
client_data_scaled = scaler.transform(client_data)

# Realizar predicciones
y_prob = model.predict_proba(client_data_scaled)[0, 1]
y_pred = model.predict(client_data_scaled)[0]

# Mostrar resultados
st.subheader("Resultado de la Predicción")
st.write(f"Probabilidad de Aprobación del Crédito: {y_prob:.2f}")
st.write("Aprobado" if y_prob>0.4253 else "Rechazado")


# Gráfico SHAP tipo waterfall
st.write("Gráfico SHAP tipo Waterfall:")
plt.figure(figsize=(6, 3))
shap_values_client = explainer.shap_values(client_data_scaled)
shap.initjs()
shap.waterfall_plot(shap.Explanation(values=shap_values_client[0], base_values=explainer.expected_value, feature_names=client_data.columns))

plt.xticks(rotation=90)
plt.xlabel('Impacto de Características en la Salida del Modelo')
plt.ylabel('Valor SHAP (impacto en la salida del modelo)')
plt.title('Importancia de Características')

# Mostrar el gráfico en Streamlit
st.pyplot(plt)

# Generar y mostrar explicación
st.subheader("Explicación del Impacto de las Características")
explicacion = generar_explicacion(feature_names, shap_values_client[0])
st.write(explicacion)

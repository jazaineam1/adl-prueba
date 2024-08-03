# app.py - Streamlit Application

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title('Análisis Exploratorio de Datos (EDA) - Entidad Bancaria')
st.write('Esta aplicación permite explorar y entender los datos de clientes y productos.')

# File uploader
uploaded_file = st.file_uploader('Cargar archivo Excel', type='xlsx')

if uploaded_file:
  # Read the Excel file
  df = pd.read_excel(uploaded_file)

# Display data preview
st.write("Vista previa de los datos:")
st.dataframe(df.head())

# Sidebar options for EDA
st.sidebar.title("Opciones de EDA")

# General information
if st.sidebar.checkbox("Mostrar información general"):
  st.subheader("Información General")
buffer = st.text(df.info())
st.text(buffer)

# Descriptive statistics
if st.sidebar.checkbox("Mostrar estadísticas descriptivas"):
  st.subheader("Estadísticas Descriptivas")
st.write(df.describe())

# Data distribution
if st.sidebar.checkbox("Visualizar distribuciones"):
  st.subheader("Distribuciones de Variables Numéricas")
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
if not num_cols.empty:
  column = st.selectbox("Seleccionar columna", num_cols)
plt.figure(figsize=(10, 4))
sns.histplot(df[column], bins=30)
st.pyplot(plt)
else:
  st.write("No hay columnas numéricas para mostrar.")

# Correlation matrix
if st.sidebar.checkbox("Mostrar matriz de correlación"):
  st.subheader("Matriz de Correlación")
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
st.pyplot(plt)

else:
  st.write("Por favor, carga un archivo Excel para comenzar.")

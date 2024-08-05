# data_processing.py - Script para el procesamiento de datos

import pandas as pd
import numpy as np

# Función para cargar y procesar los datos
def load_and_process_data(filepath):
    """
    Cargar datos desde un archivo Excel y procesarlos.
    
    Parameters:
        filepath (str): Ruta del archivo Excel.
    
    Returns:
        pd.DataFrame: DataFrame con los datos procesados.
    """
    # Cargar datos desde el archivo Excel
    df = pd.read_excel(filepath, engine='openpyxl')
    
    # Verificar si las columnas están mal formateadas
    if len(df.columns) == 1:
        # Separar los nombres de las columnas utilizando el delimitador ','
        names = df.columns[0].split(',')
        # Dividir los datos en columnas
        df = pd.DataFrame(df.iloc[:, 0].str.split(',', expand=True).values, columns=names)
    
    # Reemplazar valores de cadena vacía con NaN
    df.replace("", pd.NA, inplace=True)

    return df

# Función para convertir variables indicadoras
def convert_indicators(df):
    """
    Convertir las variables indicadoras a tipo booleano.
    
    Parameters:
        df (pd.DataFrame): DataFrame con los datos originales.
    
    Returns:
        pd.DataFrame: DataFrame con las variables indicadoras convertidas.
    """
    indicator_cols = ['housing', 'loan', 'deposit', 'tenencia_ahorros', 'tenencia_corriente', 'tenencia_cdt', 
                      'tenencia_tdc', 'tenencia_lb', 'tenencia_vehiculo']
    for col in indicator_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: True if str(x).lower() == 'yes' else False)
    return df

# Función para limpiar los datos
def clean_data(df):
    """
    Limpiar los datos en el DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame con los datos originales.
    
    Returns:
        pd.DataFrame: DataFrame con los datos limpios.
    """
    # Eliminar filas duplicadas
    df.drop_duplicates(inplace=True)
    
    # Identificar la columna de ID (primer columna) y excluirla del análisis
    id_column = df.columns[0]
    df.drop(columns=[id_column], inplace=True)
    
    # Rellenar valores faltantes con estrategias específicas
    df.fillna({
        'age': df['age'].median() if 'age' in df.columns else None,  # Rellenar edades faltantes con la mediana
        'balance': df['balance'].mean() if 'balance' in df.columns else None,  # Rellenar saldo faltante con la media
        'income': df['income'].mean() if 'income' in df.columns else 0  # Rellenar ingresos faltantes con la media
    }, inplace=True)
    
    # Convertir columnas a los tipos de datos correctos
    if 'age' in df.columns:
        df['age'] = df['age'].astype(int)
    if 'balance' in df.columns:
        df['balance'] = df['balance'].astype(float)
    if 'income' in df.columns:
        df['income'] = df['income'].astype(float)
    
    # Convertir variables categóricas a tipo 'category'
    categorical_cols = ['job', 'marital', 'education', 'default', 'contact', 'poutcome']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df

# Función principal para ejecutar el procesamiento de datos
def main():
    """
    Función principal para ejecutar el procesamiento de datos.
    """
    # Ruta del archivo de datos
    filepath = 'C:/Users/nib1l/Documents/adl-prueba/streamlit-eda-banking/data/data_prueba_ds_semisenior.xlsx'
    
    # Cargar y procesar los datos
    df = load_and_process_data(filepath)
    
    # Convertir variables indicadoras
    df = convert_indicators(df)
    
    # Limpiar los datos
    df_clean = clean_data(df)
    
    # Guardar el DataFrame procesado en un nuevo archivo Excel
    output_filepath = 'processed_data.xlsx'
    df_clean.to_excel(output_filepath, index=False)
    print(f"Datos procesados y guardados en {output_filepath}")

# Ejecutar la función principal
if __name__ == "__main__":
    main()



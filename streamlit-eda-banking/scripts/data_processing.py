import pandas as pd
import numpy as np
import os


df = pd.read_excel('C:/Users/nib1l/Downloads/data_prueba_ds_semisenior.xlsx')
names = df.columns.str.split(',')
df = df.iloc[:, 0].str.split(',', expand=True)
df.columns = names[0]
df.head()
    # data_processing.py - Script para el procesamiento de datos

# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import os

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
    df = pd.read_excel(filepath)
    
    # Separar las columnas correctamente
    names = df.columns.str.split(',')
    df = df.iloc[:, 0].str.split(',', expand=True)
    df.columns = names[0]
    
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
    
    # Rellenar valores faltantes
    df.fillna({
        'age': df['age'].median(),  # Rellenar edades faltantes con la mediana
        'balance': df['balance'].mean(),  # Rellenar saldo faltante con la media
        'income': df['income'].mean()  # Rellenar ingresos faltantes con la media
    }, inplace=True)
    
    # Convertir columnas a los tipos de datos correctos
    df['age'] = df['age'].astype(int)
    df['balance'] = df['balance'].astype(float)
    df['income'] = df['income'].astype(float)
    
    # Convertir variables categóricas a tipo 'category'
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    return df

# Función para codificar variables categóricas
def encode_categorical(df):
    """
    Codificar variables categóricas utilizando one-hot encoding.
    
    Parameters:
        df (pd.DataFrame): DataFrame con datos limpios.
    
    Returns:
        pd.DataFrame: DataFrame con variables categóricas codificadas.
    """
    # Codificación one-hot de variables categóricas
    df_encoded = pd.get_dummies(df, columns=df.select_dtypes(['category']).columns)
    return df_encoded

# Función principal para ejecutar el procesamiento de datos
def main():
    """
    Función principal para ejecutar el procesamiento de datos.
    """
    # Ruta del archivo de datos
    filepath = 'C:/Users/nib1l/Downloads/data_prueba_ds_semisenior.xlsx'
    
    # Cargar y procesar los datos
    df = load_and_process_data(filepath)
    
    # Limpiar los datos
    df_clean = clean_data(df)
    
    # Codificar variables categóricas
    df_encoded = encode_categorical(df_clean)
    
    # Guardar el DataFrame procesado en un nuevo archivo Excel
    output_filepath = 'processed_data.xlsx'
    df_encoded.to_excel(output_filepath, index=False)
    print(f"Datos procesados y guardados en {output_filepath}")

# Ejecutar la función principal
if __name__ == "__main__":
    main()

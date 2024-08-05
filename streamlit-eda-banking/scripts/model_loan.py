import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import missingno as msno 
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)

# instala imblearn.over_sampling
def load_data(filepath):
    df = pd.read_excel(filepath, engine='openpyxl')
    if len(df.columns) == 1:
        column_names = df.columns[0].split(',')
        df = pd.DataFrame(df.iloc[:, 0].str.split(',', expand=True).values, columns=column_names)
    df.replace("", np.nan, inplace=True)
    return df

file_path = r'C:\Users\nib1l\Documents\adl-prueba\streamlit-eda-banking\data\data_prueba_ds_semisenior.xlsx'
df = load_data(file_path)
df.drop(columns='',inplace=True)



# Manejar valores faltantes

# Asegurarse de que las variables sean del tipo adecuado
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
df['campaign'] = pd.to_numeric(df['campaign'], errors='coerce')
df['pdays'] = pd.to_numeric(df['pdays'], errors='coerce')
df['previous'] = pd.to_numeric(df['previous'], errors='coerce')
df['ingresos'] = pd.to_numeric(df['ingresos'], errors='coerce')
df['egresos'] = pd.to_numeric(df['egresos'], errors='coerce')
df['saldo_captacion'] = pd.to_numeric(df['saldo_captacion'], errors='coerce')
df['saldo_tdc'] = pd.to_numeric(df['saldo_tdc'], errors='coerce')
df['monto_trx_tdc'] = pd.to_numeric(df['monto_trx_tdc'], errors='coerce')
df['cantidad_trx_tdc'] = pd.to_numeric(df['cantidad_trx_tdc'], errors='coerce')
df['saldo_lb'] = pd.to_numeric(df['saldo_lb'], errors='coerce')
df['saldo_veh'] = pd.to_numeric(df['saldo_veh'], errors='coerce')
df['monto_trx_debito'] = pd.to_numeric(df['monto_trx_debito'], errors='coerce')
df['cantidad_trx_debito'] = pd.to_numeric(df['cantidad_trx_debito'], errors='coerce')

# Ajuste de características específicas
df['saldo_tdc'] = np.where(df['tenencia_cdt'] == '0.0', 0, df['saldo_tdc'])
df['monto_trx_tdc'] = np.where(df['tenencia_tdc'] == '0.0', 0, df['monto_trx_tdc'])
df['cantidad_trx_tdc'] = np.where(df['tenencia_tdc'] == '0.0', 0, df['cantidad_trx_tdc'])
df['saldo_lb'] = np.where(df['tenencia_lb'] == '0.0', 0, df['saldo_lb'])
df['saldo_veh'] = np.where(df['tenencia_vehiculo'] == '0.0', 0, df['saldo_veh'])
df['monto_trx_debito'] = np.where(df['deposit'] == '0.0', 0, df['monto_trx_debito'])
df['cantidad_trx_debito'] = np.where(df['deposit'] == '0.0', 0, df['cantidad_trx_debito'])
categorical_columns_with_unknown = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
    'poutcome', 'deposit', 'tenencia_ahorros', 'tenencia_corriente', 
    'tenencia_cdt', 'tenencia_tdc', 'tenencia_lb', 'tenencia_vehiculo'
]

# Reemplazar NaN con 'unknown' en las columnas categóricas
df[categorical_columns_with_unknown] = df[categorical_columns_with_unknown].fillna('unknown')

# Ajuste de características específicas para la variable 'tenencia'
df['tenencia_cdt'] = np.where(df['saldo_tdc'] > 0, '1.0', df['tenencia_cdt'])
df['tenencia_tdc'] = np.where((df['monto_trx_tdc'] > 0) | (df['cantidad_trx_tdc'] > 0), '1.0', df['tenencia_tdc'])
df['tenencia_lb'] = np.where(df['saldo_lb'] > 0, '1.0', df['tenencia_lb'])
df['tenencia_vehiculo'] = np.where(df['saldo_veh'] > 0, '1.0', df['tenencia_vehiculo'])
df['deposit'] = np.where((df['monto_trx_debito'] > 0) | (df['cantidad_trx_debito'] > 0), '1.0', df['deposit'])


def replace_negatives_with_na(df, columns_to_round):
    """
    Reemplaza los valores negativos en las columnas especificadas con NA.

    :param df: DataFrame de pandas
    :param columns_to_round: Lista de nombres de columnas a procesar
    :return: DataFrame con valores negativos reemplazados por NA
    """
    df_copy = df.copy()
    for column in columns_to_round:
        df_copy[column] = df_copy[column].apply(lambda x: np.nan if x < 0 else x)
    return df_copy



columns_to_round = ['cantidad_trx_tdc', 'cantidad_trx_debito']

# Redondear las variables especificadas a números enteros
df[columns_to_round] = df[columns_to_round].round(0)
df = replace_negatives_with_na(df, columns_to_round)
df.info()
porcentaje_nulos = df.isnull().mean() * 100
columnas_menos_nulos = porcentaje_nulos[porcentaje_nulos < 50].index
df = df[columnas_menos_nulos]

msno.bar(df)
msno.matrix(df)
# plt.show()
#df = df.dropna()
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['number']).columns

# Aplicar get_dummies solo a las variables categóricas
X_categorical = pd.get_dummies(df[categorical_columns], drop_first=True)

# Mantener las variables numéricas
X_numerical = df[numerical_columns]

# Concatenar los datos categóricos y numéricos
X = pd.concat([X_numerical, X_categorical], axis=1)
y_loan = X['loan_1'].astype('int')
# Asegurarse de eliminar la columna de la etiqueta si está presente en X
X = X.drop(columns=['loan_1'], errors='ignore')

for i in df.select_dtypes('O').columns:
  print(i)
  df[i].unique()



# Escalar las características numéricas
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_loan_train, y_loan_test = train_test_split(
    X, y_loan, test_size=0.2, random_state=42
)


# Modelo XGBoost con regularización y otros ajustes para reducir el sobreajuste
xgb = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=len(y_loan_train) / sum(y_loan_train == 1),
    max_depth=3,  # Profundidad máxima de los árboles
    n_estimators=100,  # Número de árboles
    learning_rate=0.01,  # Tasa de aprendizaje
    subsample=0.8,  # Submuestreo de ejemplos
    colsample_bytree=0.8,  # Submuestreo de características
    reg_alpha=0.1,  # Regularización L1
    reg_lambda=0.1  # Regularización L2
)

# Ajuste de Hiperparámetros para XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    #'reg_alpha': [0.01, 0.1, 1],
    #'reg_lambda': [0.01, 0.1, 1]
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_loan_train)

# Mejor modelo y parámetros encontrados
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Mejores parámetros encontrados:", best_params)

# Evaluar el mejor modelo en el conjunto de prueba
best_model_predictions = best_model.predict(X_test)
best_model_accuracy = accuracy_score(y_loan_test, best_model_predictions)
print(f"Precisión del mejor modelo en el conjunto de prueba: {best_model_accuracy}")

# Reporte de clasificación y matriz de confusión
print(classification_report(y_loan_test, best_model_predictions))
print(confusion_matrix(y_loan_test, best_model_predictions))

y_probs = best_model.predict_proba(X_test)[:, 1]

# Ajustar el umbral de decisión (por ejemplo, 0.3)
threshold = 0.3
y_pred_adjusted = (y_probs >= threshold).astype(int)

# Calcular el nuevo classification report
print(classification_report(y_loan_test, y_pred_adjusted))


# Obtener las probabilidades predichas para la clase positiva
y_probs = best_model.predict_proba(X_test)[:, 1]

# Calcular la curva precision-recall y la curva ROC
precision, recall, thresholds_pr = precision_recall_curve(y_loan_test, y_probs)
fpr, tpr, thresholds_roc = roc_curve(y_loan_test, y_probs)

# Seleccionar el umbral con el mayor F1-score
f1_scores = 2 * precision * recall / (precision + recall)
best_threshold = thresholds_pr[np.argmax(f1_scores)]

print(f"Best threshold based on F1-score: {best_threshold}")

# Evaluar el modelo con el umbral seleccionado
y_pred_adjusted = (y_probs >= best_threshold).astype(int)
print(classification_report(y_loan_test, y_pred_adjusted))



importances = best_model.feature_importances_
feature_names = X_categorical.columns.tolist() + X_numerical.columns.tolist()
feature_names.remove('loan_1') 
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(12, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.title('Importancia de las características en el modelo XGBoost')
plt.gca().invert_yaxis()
plt.show()
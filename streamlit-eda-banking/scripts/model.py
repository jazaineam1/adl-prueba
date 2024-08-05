import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# Cargar los datos
def load_data(filepath):
    df = pd.read_excel(filepath, engine='openpyxl')
    if len(df.columns) == 1:
        column_names = df.columns[0].split(',')
        df = pd.DataFrame(df.iloc[:, 0].str.split(',', expand=True).values, columns=column_names)
    df.replace("", np.nan, inplace=True)
    return df

file_path = r'C:\Users\nib1l\Documents\adl-prueba\streamlit-eda-banking\data\data_prueba_ds_semisenior.xlsx'
df = load_data(file_path)
df = df.drop(columns='')
#df =df.dropna(subset='balance')

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

## haz que tenencia_cdt=0 entonces saldo_cdt= 0     y monto_trx_tdc=0 y cantidad_trx_tdc=0      
df['saldo_tdc'] = np.where(df['tenencia_cdt'] == '0', 0, df['saldo_tdc'])
df['monto_trx_tdc'] = np.where(df['tenencia_tdc'] == '0', 0, df['monto_trx_tdc'])
df['cantidad_trx_tdc'] = np.where(df['tenencia_tdc'] == '0', 0, df['cantidad_trx_tdc'])

# haz lo mismo anterior con  tenencia_lb con saldo_lb y tenencia_vehiculo con saldo_veh, deposito con la pareja monto_trx_debito y cantidad_trx_debito 
df['saldo_lb'] = np.where(df['tenencia_lb'] == '0', 0, df['saldo_lb'])
df['saldo_veh'] = np.where(df['tenencia_vehiculo'] == '0', 0, df['saldo_veh'])
df['monto_trx_debito'] = np.where(df['deposit'] == '0', 0, df['monto_trx_debito'])
df['cantidad_trx_debito'] = np.where(df['deposit'] == '0', 0, df['cantidad_trx_debito'])

# ver nan es df
print(df.isnull().sum())

df.info()
# Manejo de NaN y reemplazo por 'unknown'
columns_to_replace = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome',
                      'deposit', 'tenencia_ahorros', 'tenencia_corriente', 'tenencia_cdt',
                      'tenencia_tdc', 'tenencia_lb', 'tenencia_vehiculo']
df[columns_to_replace] = df[columns_to_replace].fillna('unknown')

# Asegúrate de que todas las columnas categóricas estén codificadas numéricamente
X = df.drop(columns=['housing', 'loan'])
X = pd.get_dummies(X)
y_housing = df['housing'].astype('int')
y_loan = df['loan'].astype('int')

# Escalar las características numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_housing_train, y_housing_test = train_test_split(X_scaled, y_housing, test_size=0.2, random_state=42)

# Definir el modelo XGBoost
xgb = XGBClassifier()

# Ajuste de Hiperparámetros para XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}

# Búsqueda de hiperparámetros
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_housing_train)

# Mejor modelo y parámetros encontrados
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Mejores parámetros encontrados:", best_params)

# Evaluar el mejor modelo en el conjunto de prueba
best_model_predictions = best_model.predict(X_test)
best_model_accuracy = accuracy_score(y_housing_test, best_model_predictions)
print(f"Precisión del mejor modelo en el conjunto de prueba: {best_model_accuracy}")

## muestrame todos los valores recall, f1 score, precision, accuracy, roc_auc_score
from sklearn.metrics import classification_report
print(classification_report(y_housing_test, best_model_predictions))

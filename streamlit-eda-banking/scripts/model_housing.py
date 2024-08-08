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

# Función para cargar datos
def load_data(filepath):
    df = pd.read_excel(filepath, engine='openpyxl')
    if len(df.columns) == 1:
        column_names = df.columns[0].split(',')
        df = pd.DataFrame(df.iloc[:, 0].str.split(',', expand=True).values, columns=column_names)
    df.replace("", np.nan, inplace=True)
    return df



df['deposit'] = df['deposit'].astype('str')
numeric_columns = [
    'age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'ingresos', 
    'egresos', 'saldo_captacion', 'saldo_tdc', 'monto_trx_tdc', 'cantidad_trx_tdc',
    'saldo_lb', 'saldo_veh', 'monto_trx_debito', 'cantidad_trx_debito'
]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

df['tenencia_cdt'] = np.where(df['saldo_tdc'].notnull(), '1.0', df['tenencia_cdt'])
df['monto_trx_tdc'] = np.where(df['tenencia_tdc'] == '0.0', 0, df['monto_trx_tdc'])
df['cantidad_trx_tdc'] = np.where(df['tenencia_tdc'] == '0.0', 0, df['cantidad_trx_tdc'])
df['saldo_lb'] = np.where(df['tenencia_lb'] == '0.0', 0, df['saldo_lb'])
df['saldo_veh'] = np.where(df['tenencia_vehiculo'] == '0.0', 0, df['saldo_veh'])
df['monto_trx_debito'] = np.where(df['deposit'] == '0', 0, df['monto_trx_debito'])
df['cantidad_trx_debito'] = np.where(df['deposit'] == '0', 0, df['cantidad_trx_debito'])
df['tenencia_vehiculo'] = np.where(df['saldo_veh'] == '0', 0, df['tenencia_vehiculo'])
df['tenencia_lb'] = np.where(df['saldo_lb'] == '0', 0, df['tenencia_lb'])
df['tenencia_tdc'] = np.where((~df['saldo_tdc'].isnull())|(~df['monto_trx_tdc'].isnull())|(~df['cantidad_trx_tdc'].isnull()), 0, df['tenencia_tdc'])

df.dropna(subset=['tenencia_tdc', 'tenencia_lb', 'tenencia_vehiculo'], inplace=True)
columns_to_check = [
    'ingresos', 'egresos', 'saldo_captacion', 'saldo_tdc', 
    'monto_trx_tdc', 'cantidad_trx_tdc', 'saldo_lb', 'saldo_veh'
]
df = df[~(df[columns_to_check] < 0).any(axis=1)]
df.reset_index(drop=True, inplace=True)

categorical_columns_with_unknown = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
    'poutcome', 'deposit', 'tenencia_ahorros', 'tenencia_corriente', 
    'tenencia_cdt', 'tenencia_tdc', 'tenencia_lb', 'tenencia_vehiculo'
]
df[categorical_columns_with_unknown] = df[categorical_columns_with_unknown].fillna('unknown')

porcentaje_nulos = df.isnull().mean() * 100
columnas_menos_nulos = porcentaje_nulos[porcentaje_nulos < 35].index
df = df[columnas_menos_nulos]
df.drop(columns=['monto_trx_debito', 'cantidad_trx_debito'], inplace=True)
df = df[~(df[columns_to_check] < 0).any(axis=1)]
numeric_columns =  [
    'age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'ingresos', 
    'egresos', 'saldo_captacion', 'saldo_tdc', 'monto_trx_tdc', 'cantidad_trx_tdc',
    'saldo_lb', 'saldo_veh']
df_numeric = df[numeric_columns]

categorical_columns = df.select_dtypes(include=['object']).columns
X_categorical = pd.get_dummies(df[categorical_columns], drop_first=True)
X_numerical = df[numeric_columns]

X = pd.concat([X_numerical, X_categorical], axis=1)
y_housing = df['housing'].astype('int')

X = X.drop(columns=['loan_1','housing_1'], errors='ignore')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_housing_train, y_housing_test = train_test_split(X_scaled, y_housing, test_size=0.2, random_state=42)

xgb = XGBClassifier()

param_grid = {
    'n_estimators': [50,100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_housing_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Mejores parámetros encontrados:", best_params)
y_probs_xgb = best_model.predict_proba(X_test)[:, 1]
best_model_predictions = best_model.predict(X_test)
best_model_accuracy = accuracy_score(y_housing_test, best_model_predictions)
print(f"Precisión del mejor modelo en el conjunto de prueba: {best_model_accuracy}")

from sklearn.metrics import classification_report
print(classification_report(y_housing_test, best_model_predictions))

from sklearn.metrics import roc_curve, precision_recall_curve
precision, recall, thresholds_pr = precision_recall_curve(y_housing_test, y_probs_xgb)
fpr, tpr, thresholds_roc = roc_curve(y_housing_test, best_model_predictions)

f1_scores = 2 * precision * recall / (precision + recall)
best_threshold = thresholds_pr[np.argmax(f1_scores)]

print(f"Best threshold based on F1-score: {best_threshold}")

y_pred_adjusted = (y_probs_xgb >= best_threshold).astype(int)
print(classification_report(y_housing_test, y_pred_adjusted))

from sklearn.metrics import roc_auc_score

y_probs = best_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_housing_test, y_probs)
roc_auc


import shap
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns)

shap.dependence_plot("balance", shap_values, X_test, feature_names=X.columns)
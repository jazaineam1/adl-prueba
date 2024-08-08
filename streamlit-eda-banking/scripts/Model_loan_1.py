import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, roc_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.metrics import confusion_matrix
import warnings
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore", category=FutureWarning)

def load_data(filepath):
    df = pd.read_excel(filepath, engine='openpyxl')
    if len(df.columns) == 1:
        column_names = df.columns[0].split(',')
        df = pd.DataFrame(df.iloc[:, 0].str.split(',', expand=True).values, columns=column_names)
    df.replace("", np.nan, inplace=True)
    return df

file_path = r'C:\Users\nib1l\Documents\adl-prueba\streamlit-eda-banking\data\data_prueba_ds_semisenior.xlsx'
df = load_data(file_path)

df.drop(columns='', inplace=True)
df['deposit'] = df['deposit'].astype('str')
df['deposit'].value_counts()

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

df.reset_index(drop=True, inplace=True)

categorical_columns_with_unknown = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
    'poutcome', 'deposit', 'tenencia_ahorros', 'tenencia_corriente', 
    'tenencia_cdt', 'tenencia_tdc', 'tenencia_lb', 'tenencia_vehiculo'
]
df[categorical_columns_with_unknown] = df[categorical_columns_with_unknown].fillna('unknown')
df.isna().sum()
porcentaje_nulos = df.isnull().mean() * 100
columnas_menos_nulos = porcentaje_nulos[porcentaje_nulos < 35].index
df = df[columnas_menos_nulos]
df.drop(columns=['monto_trx_debito', 'cantidad_trx_debito'],inplace=True)
df.info()
columns_to_check = [
    'ingresos', 'egresos', 'saldo_captacion', 'saldo_tdc', 
    'monto_trx_tdc', 'cantidad_trx_tdc', 'saldo_lb', 'saldo_veh'
]





print(df.isnull().sum())
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['number']).columns

X_categorical = pd.get_dummies(df[categorical_columns], drop_first=True)
X_numerical = df[numerical_columns]

X = pd.concat([X_numerical, X_categorical], axis=1)
y_loan = df['loan'].astype('int')

X = X.drop(columns=['loan_1'], errors='ignore')

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_loan_train, y_loan_test = train_test_split(
    X, y_loan, test_size=0.2, random_state=42
)


params = {


    "XGBoost": {
        "n_estimators": [20,50, 150],  
        "max_depth": [3, 7],
        "learning_rate": [0.01, 0.2],
          "scale_pos_weight": [24, 36],
    }
}


models = {
    "XGBoost": XGBClassifier(random_state=42,  eval_metric='logloss'),
}

results = {}
for model_name, model in models.items():
    clf = GridSearchCV(model, params[model_name], cv=5, scoring='f1',verbose=2,n_jobs=-1)
    clf.fit(X_train, y_loan_train)
    best_model = clf.best_estimator_
    
    y_probs = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_loan_test, y_pred)
    report = classification_report(y_loan_test, y_pred)
    confusion = confusion_matrix(y_loan_test, y_pred)
    
    results[model_name] = {
        "best_model": best_model,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": confusion
    }
1;1


for model_name, metrics in results.items():
    print(f"Resultados para {model_name}:")
    print(f"Mejor modelo: {metrics['best_model']}")
    print(f"Precisión: {metrics['accuracy']}")
    print(f"Reporte de clasificación:\n{metrics['classification_report']}")
    print(f"Matriz de confusión:\n{metrics['confusion_matrix']}")
    print("-" * 50)


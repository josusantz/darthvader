import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import warnings       
import pickle
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import xgboost as xgb
from mytools import encontrar_estacao, cortar_serie_temporal, remover_outliers
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('dados_A215_H_2008-06-13_2024-01-01.csv',sep= ';', header = 9)
df['Data Medicao'] = pd.to_datetime(df['Data Medicao'])
df['Hora Medicao'] = pd.to_datetime(df['Hora Medicao'])
df['ano'] = df['Data Medicao'].dt.year
df['mes'] = df['Data Medicao'].dt.month
df['dia'] = df['Data Medicao'].dt.day
# df['hora']= df['Hora Medicao'].str[-4:-2].astype(int)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.').astype(float)


df['Hora Medicao'] = df['Hora Medicao'].astype(str)
df['hora'] = df['Hora Medicao'].str[-4:-2].astype(int)
df = df.drop(['Hora Medicao'],axis=1)
df['hora'] = pd.to_timedelta(df['hora'], unit='h') - pd.Timedelta(hours=3)
df['Data_Hora'] = df['Data Medicao'] + df['hora']
df.set_index('Data_Hora', inplace=True)
df['Data_Hora'] = df['Data Medicao'] + df['hora']
df = df.drop('Unnamed: 22', axis =1)
df['estacao'] = df.apply(encontrar_estacao, axis=1)

df = df.dropna()

df_crop = cortar_serie_temporal(df = df,data_inicio = '2010-01-01', data_fim = '2020-12-31')

df = remover_outliers(df_crop)
df.to_csv('cropped.csv')
scaler = MinMaxScaler()

features = ['TEMPERATURA DO PONTO DE ORVALHO(°C)','VENTO, VELOCIDADE HORARIA(m/s)', 'UMIDADE RELATIVA DO AR, HORARIA(%)', 'TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)']
target = ['RADIACAO GLOBAL(Kj/m²)']
dataframe = df
X = dataframe[features].values
y = dataframe[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#fim do pre-processamento 
print("Pre-processamento finalizado")
# Definindo o modelo SVR
svr = SVR()
# Definindo os parâmetros para GridSearchCV
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5]
}

# Realizando a busca em grade para encontrar os melhores parâmetros
grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_svr.fit(X_train_scaled, y_train)

# Melhores parâmetros encontrados
print("Melhores parâmetros para SVR:", grid_search_svr.best_params_)

# Avaliação do modelo
svr_best = grid_search_svr.best_estimator_
svr_best.fit(X_train_scaled, y_train)
svr_pred = svr_best.predict(X_test_scaled)


print("svr finalizado")
# Definindo o modelo Random Forest
rf = RandomForestRegressor(random_state=42)

# Definindo os parâmetros para GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Realizando a busca em grade para encontrar os melhores parâmetros
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

# Melhores parâmetros encontrados
print("Melhores parâmetros para Random Forest:", grid_search_rf.best_params_)

# Avaliação do modelo
rf_best = grid_search_rf.best_estimator_
rf_best.fit(X_train, y_train)
rf_pred = rf_best.predict(X_test)


print("random florest finalizado")
# Convertendo os dados para DMatrix para uso com XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Definindo os parâmetros para XGBoost (exemplo)
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Treinamento do modelo XGBoost
num_round = 100
xgb_model = xgb.train(params, dtrain, num_round)

# Avaliação do modelo
xgb_pred = xgb_model.predict(dtest)

# Construindo o modelo de RNA
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Camada de saída

# Compilando o modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Treinamento do modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

# Avaliação do modelo
loss = model.evaluate(X_test_scaled, y_test)
print("Erro de teste (MSE):", loss)

# Previsões
rna_pred = model.predict(X_test_scaled).flatten()

# Avaliação dos modelos
# print("SVR - MSE:", mean_squared_error(y_test, svr_pred))
# print("Random Forest - MSE:", mean_squared_error(y_test, rf_pred))
# print("XGBoost - MSE:", mean_squared_error(y_test, xgb_pred))
# print("RNA - MSE:", mean_squared_error(y_test, rna_pred))

# print("SVR - R²:", r2_score(y_test, svr_pred))
# print("Random Forest - R²:", r2_score(y_test, rf_pred))
# print("XGBoost - R²:", r2_score(y_test, xgb_pred))
# print("RNA - R²:", r2_score(y_test, rna_pred))


# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Dictionary to store evaluation metrics
evaluation_metrics = {}

# SVR Evaluation
evaluation_metrics['SVR'] = {
    'MSE': mean_squared_error(y_test, svr_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, svr_pred)),
    'MAE': mean_absolute_error(y_test, svr_pred),
    'MAPE': mean_absolute_percentage_error(y_test, svr_pred),
    'R²': r2_score(y_test, svr_pred),
    'EVS': explained_variance_score(y_test, svr_pred)
}

# Random Forest Evaluation
evaluation_metrics['Random Forest'] = {
    'MSE': mean_squared_error(y_test, rf_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
    'MAE': mean_absolute_error(y_test, rf_pred),
    'MAPE': mean_absolute_percentage_error(y_test, rf_pred),
    'R²': r2_score(y_test, rf_pred),
    'EVS': explained_variance_score(y_test, rf_pred)
}

# XGBoost Evaluation
evaluation_metrics['XGBoost'] = {
    'MSE': mean_squared_error(y_test, xgb_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
    'MAE': mean_absolute_error(y_test, xgb_pred),
    'MAPE': mean_absolute_percentage_error(y_test, xgb_pred),
    'R²': r2_score(y_test, xgb_pred),
    'EVS': explained_variance_score(y_test, xgb_pred)
}

# RNA Evaluation
evaluation_metrics['RNA'] = {
    'MSE': mean_squared_error(y_test, rna_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, rna_pred)),
    'MAE': mean_absolute_error(y_test, rna_pred),
    'MAPE': mean_absolute_percentage_error(y_test, rna_pred),
    'R²': r2_score(y_test, rna_pred),
    'EVS': explained_variance_score(y_test, rna_pred)
}

# Print evaluation metrics for comparison
for model_name, metrics in evaluation_metrics.items():
    print(f"{model_name} - MSE: {metrics['MSE']}, RMSE: {metrics['RMSE']}, MAE: {metrics['MAE']}, MAPE: {metrics['MAPE']}, R²: {metrics['R²']}, EVS: {metrics['EVS']}")


#Save SVR, Random Forest, and XGBoost models using pickle
models = {'svr_best': svr_best, 'rf_best': rf_best}
for name, model in models.items():
    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(model, file)
        
xgb_model.save_model('xgb_model.json')
model.save('rna_model.keras')

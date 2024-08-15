import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings       
import pickle
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")
from mytools import encontrar_estacao, cortar_serie_temporal, remover_outliers
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('clean_data.csv')
scaler = MinMaxScaler()

features = ['TEMPERATURA DO PONTO DE ORVALHO(°C)','VENTO, VELOCIDADE HORARIA(m/s)', 'UMIDADE RELATIVA DO AR, HORARIA(%)', 'TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)']
target = ['RADIACAO GLOBAL(Kj/m²)']
dataframe = df
X = dataframe[features].values
y = dataframe[target].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR()

param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5]
}

grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
print('iniciado grid search..')
grid_search_svr.fit(X_train_scaled, y_train)

print("Melhores parâmetros para SVR:", grid_search_svr.best_params_)

svr_best = grid_search_svr.best_estimator_
print('treinando svr..')
svr_best.fit(X_train_scaled, y_train)
svr_pred = svr_best.predict(X_test_scaled)

with open('SVR_model-01.pkl', 'wb') as file: 
    pickle.dump(svr_best, file)

with open('SVR_model.pkl', 'rb') as file: 
    SVR_loaded_model = pickle.load(file) 


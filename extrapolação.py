import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
data = 'cropped.csv'
df = pd.read_csv(data)

print(f"dataframe '{data}' is loaded.")

scaler = MinMaxScaler()
features = ['TEMPERATURA DO PONTO DE ORVALHO(°C)','VENTO, VELOCIDADE HORARIA(m/s)', 'UMIDADE RELATIVA DO AR, HORARIA(%)', 'TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)']
target = ['RADIACAO GLOBAL(Kj/m²)']
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('train-test split complete')
with open('SVR_model.pkl', 'rb') as file: 
    SVR_loaded_model = pkl.load(file) 
print("model 'SVR' is loaded")
print('predicting...')
predict = SVR_loaded_model.predict(X_test_scaled)
solar_irradiation = {'Solar_Irradiation':predict,'Features':
print(predict)

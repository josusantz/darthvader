from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)

# Carregar o modelo serializado
with open('SVR_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Bem-vindo ao servidor de previsão de irradiação solar!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify(irradiação_solar=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {
    'features': [25.0, 60.0, 5.0, 20.0]
}

headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.json())

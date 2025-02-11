import joblib
import numpy as np

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([7.527542076, 7.41045765,	1.443571925, 1.540246725, 0.80915767, 0.617950857, 0.245482773, 0.382611543, 2.430181503])
    prediction = model.predict(X_test.reshape(1, -1))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)
from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load('model.joblib')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_strength():
    cement = float(request.form.get('cement'))
    blast_furnace_slag = float(request.form.get('blast_furnace_slag'))
    fly_ash = float(request.form.get('fly_ash'))
    water = float(request.form.get('water'))
    superplasticizer = float(request.form.get('superplasticizer'))
    coarse_aggregate = float(request.form.get('coarse_aggregate'))
    fine_aggregate = float(request.form.get('fine_aggregate'))
    age = int(request.form.get('age'))

    # Prediction
    result = model.predict(np.array([cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]).reshape(1, 8))

    return str(result)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('GB_Model.pkl')

def validate_input(data):
    # Validate input data
    if not all(isinstance(val, (int, float)) for val in data):
        return False
    if len(data) != 8:
        return False
    return True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input data from form
            pregnancies = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            blood_pressure = int(request.form['blood_pressure'])
            skin_thickness = int(request.form['skin_thickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])

            # Validate input data
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            if not validate_input(input_data):
                raise ValueError("Invalid input data")

            # Make prediction
            data = np.array([input_data])
            prediction = model.predict(data)

            if prediction[0] == 1:
                result = 'Diabetic'
            else:
                result = 'Non-Diabetic'

            return render_template('result.html', result=result)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('error.html', error=error_message)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if request.method == 'POST':
        try:
            # Get input data from JSON
            data = request.json
            if not validate_input(data):
                raise ValueError("Invalid input data")

            # Make prediction
            data = np.array([list(data.values())])
            prediction = model.predict(data)

            if prediction[0] == 1:
                result = 'Diabetic'
            else:
                result = 'Non-Diabetic'

            return jsonify({'result': result})

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

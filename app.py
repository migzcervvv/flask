from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

# Load the model and scaler once when the server starts
model = load_model('smog_prediction_model.keras')
input_scaler = joblib.load('input_scaler.gz')
output_scaler = joblib.load('output_scaler.gz')

def load_and_predict(input_data):
    # Scale the input data using the input scaler
    input_data_scaled = input_scaler.transform(input_data.reshape(1, -1))
    input_data_scaled = input_data_scaled.reshape(1, 1, -1)

    # Predict using the model
    prediction = model.predict(input_data_scaled)
    
    # Inverse transform the prediction using the output scaler
    inversed_prediction = output_scaler.inverse_transform(prediction)
    
    # Formatting the output as JSON
    labels = ["smog_percentage", "wind_direction", "wind_speed"]
    result = {}
    for i in range(3):
        day_index = i + 1
        result[f"day_{day_index}"] = {
            labels[j]: float(inversed_prediction[0, i*3 + j]) for j in range(3)
        }
    return json.dumps(result)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        predefined_data = np.array([[30.5, 22, 78, 0.6, 11]])  # Example feature set
        result = load_and_predict(predefined_data)  # Call the prediction function
        return jsonify(result), 200  # Convert the result into JSON format and return
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5140)

    
"""
{
"range": "Sheet1!A2:E97"
}

flask run --host 0.0.0.0 --port 5140
"""
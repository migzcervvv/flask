from flask import Flask, json, request, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from validate_email_address import validate_email
from flask_cors import CORS
import os
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db
import re
from collections import defaultdict

load_dotenv

app = Flask(__name__)

# Initialize the app with a service account, granting admin privileges
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://metrobreathe-may8-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

CORS(app)
creds_path = os.path.join(os.path.dirname(__file__), 'creds.json')
creds = service_account.Credentials.from_service_account_file(creds_path)

# Load the model and scaler
model = load_model('smog_prediction_model.keras')
input_scaler = joblib.load('input_scaler.gz')
output_scaler = joblib.load('output_scaler.gz')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Gmail account credentials
gmail_user = os.getenv('GMAIL_USER')
gmail_password = os.getenv('GMAIL_PASSWORD')

@app.route('/submit-form', methods=['POST'])
def receive_form():
    try:
        data = request.json
        name = data['name']
        email = data['email']
        message = data['message']

        if not validate_email(email):
            return jsonify({'message': 'Invalid email address'}), 400

        sender_email = email
        receiver_email = gmail_user

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = 'MetroBreathe Email'
        body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(gmail_user, gmail_password)
            smtp.sendmail(sender_email, receiver_email, msg.as_string())

        return jsonify({'message': 'Form received and processed successfully'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

def extract_numeric_values(data):
    numeric_data = {}
    pattern = re.compile(r'\d+\.\d+|\d+')

    for parameter, content in data.items():
        values = content.get('Value', {})
        latest_entry = list(values.values())[-1]  # Get the latest entry
        match = pattern.search(latest_entry)
        numeric_data[parameter] = float(match.group()) if match else np.nan

    return numeric_data

def preprocess_data(numeric_data):
    """Preprocess the data to match the expected format and calculate RollingMean_PM25."""
    # Convert dictionary to DataFrame
    df = pd.DataFrame([numeric_data])  # Create a DataFrame with a single row

    df['RollingMean_PM25'] = df['PM25']  # Simplistic approximation if no past data

    # Ensure we have the right columns in the right order
    expected_features = ['RollingMean_PM25', 'Temperature', 'Humidity', 'CO (Carbon Monoxide)', 'O3 (Ozone)', 'PM10', 'PM25', 'VOC (Volatile Organic Compounds)']
    for feature in expected_features:
        if feature not in df:
            df[feature] = np.nan

    # Scale the data
    scaled_data = input_scaler.transform(df[expected_features])

    # Reshape for LSTM input
    scaled_data = scaled_data.reshape(1, 1, len(expected_features))  # Shape (1, 1, number of features)
    return scaled_data

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Fetch data from Firebase
        ref = db.reference('/')
        firebase_data = ref.get()

        # Extract numeric values from the nested structure
        numeric_data = extract_numeric_values(firebase_data)
        
        # Preprocess the fetched data
        data_preprocessed = preprocess_data(numeric_data)

        # Make prediction
        prediction = model.predict(data_preprocessed)
        prediction = output_scaler.inverse_transform(prediction)  # Inverse transform if your output is scaled

        # Convert numpy float32 to native Python float for JSON serialization
        prediction = prediction.flatten().tolist()  # Flatten and convert to list
        prediction = [float(num) for num in prediction]  # Convert elements to float

        # Format the prediction result
        result = {
            "day_1": {"smog_percentage": prediction[0]},
            "day_2": {"smog_percentage": prediction[1]},
            "day_3": {"smog_percentage": prediction[2]}
        }

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)

    #curl http://127.0.0.1:10000/predict
from flask import Flask, json, request, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from validate_email_address import validate_email
from flask_cors import CORS

from google.oauth2 import service_account
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Importing forecasting functions
from forecasting import load_and_preprocess_data, predict_and_format_output, prepare_output_scaler

app = Flask(__name__)
CORS(app)
creds = service_account.Credentials.from_service_account_file('creds.json')

# Load the model and scaler once when the server starts
model = load_model('smog_prediction_model.keras')
input_scaler = joblib.load('input_scaler.gz')
output_scaler = joblib.load('output_scaler.gz')

# Gmail account credentials
gmail_user = "miggzc1@gmail.com"
gmail_password = "zzek ptpi hkzf kysa"

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

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Load new data
        data, _ = load_and_preprocess_data('testdata.csv')  
        result = predict_and_format_output(model, data[-1:, :, :], output_scaler)  # Predict the latest data point
        return jsonify(json.loads(result)), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

    
"""
{
"range": "Sheet1!A2:E97"
}

flask run --host 0.0.0.0 --port 5140
"""
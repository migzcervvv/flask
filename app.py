from flask import Flask, request, jsonify
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json
from forecasting import load_and_predict  # Assuming this function is in forecasting.py

app = Flask(__name__)
creds = service_account.Credentials.from_service_account_file('creds.json')

@app.route("/readSheets", methods=['POST'])
def readSheets_API():
    param = request.get_json()
    sheet_range = param['range']
    return readGsheet(sheet_range)

def readGsheet(sheet_range):
    SPREADSHEET_ID = "1QY_I_7ci1pZkraeUopbVrYgI-kphveqrOKsGq890Z_w"
    service = build("sheets", "v4", credentials=creds)
    sheetread = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=sheet_range).execute()
    values = sheetread.get('values', [])
    df = pd.DataFrame(values, columns=['PM2.5', 'PM10', 'VOCS', 'CO', 'O3'])
    df = df.apply(pd.to_numeric, errors='coerce')
    return jsonify(df.to_dict(orient='records'))

@app.route('/predict', methods=['GET'])
def predict():
    try:
        predefined_data = np.array([[30.5, 22, 78, 0.6, 11]])  # Example feature set
        predefined_data = predefined_data.reshape(1, 1, -1)  # Reshape for LSTM model: (samples, timesteps, features)

        # Call the prediction function
        result = load_and_predict(predefined_data)
        # Convert the result into JSON format
        return jsonify(result), 200
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
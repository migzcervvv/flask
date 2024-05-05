from flask import Flask, request, jsonify
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from forecasting import train_and_predict  # Ensure this import is correct

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
def predict_smog():
    df = fetch_data_from_google_sheets()
    if df.shape[0] < 24:
        return jsonify({"error": "Not enough data to make prediction. Need at least 24 data points."})
    predictions = train_and_predict(df, future_steps=3)  # Use the function directly
    return jsonify(predictions.tolist())

def fetch_data_from_google_sheets():
    sheet = build('sheets', 'v4', credentials=creds).spreadsheets()
    result = sheet.values().get(spreadsheetId='1QY_I_7ci1pZkraeUopbVrYgI-kphveqrOKsGq890Z_w', range='Sheet1!A2:E97').execute()
    data = result.get('values', [])
    df = pd.DataFrame(data, columns=['PM2.5', 'PM10', 'VOCS', 'CO', 'O3'])
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5140)
    
"""
{
"range": "Sheet1!A2:E97"
}

flask run --host 0.0.0.0 --port 5140
"""
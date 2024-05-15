import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import joblib
from dotenv import load_dotenv

load_dotenv() 

def load_and_preprocess_data(data):
    # Assuming 'data' is a DataFrame or path to CSV
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data)

    data['RollingMean_PM25'] = data['PM25'].rolling(window=3).mean().fillna(method='bfill')
    features = ['RollingMean_PM25', 'Temperature', 'Humidity', 'CO (Carbon Monoxide)', 'O3 (Ozone)', 'PM10', 'PM25', 'VOC (Volatile Organic Compounds)']
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[features])

    # Correctly reshape data to (samples, timesteps, features)
    data_scaled = data_scaled.reshape(data_scaled.shape[0], 1, len(features))
    return data_scaled, scaler

def create_advanced_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, input_shape=input_shape)),  
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(3)  # Assuming the output layer size is correct for your task
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=64)
    return model

def prepare_targets(data, forecast_horizon=3):
    targets = np.zeros((data.shape[0] - forecast_horizon, forecast_horizon))
    for i in range(forecast_horizon):
        targets[:, i] = data[i+1:data.shape[0]-forecast_horizon+i+1, 0, 0]  # Only PM25
    return targets

def prepare_output_scaler(targets):
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    output_scaler.fit(targets)
    return output_scaler

def predict_and_format_output(model, data, output_scaler):
    prediction = model.predict(data)
    if prediction.shape[1] != 3:  # Reshape check for 3-day forecast
        prediction = prediction.reshape(1, 3)
    inversed_prediction = output_scaler.inverse_transform(prediction)

    result = {}
    label = "smog_percentage"
    for i in range(3):
        day_index = i + 1
        result[f"day_{day_index}"] = {label: float(inversed_prediction[0, i])}
    return json.dumps(result)

def save_model_and_scaler(model, scaler, model_filename, scaler_filename):
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)

def main():
    filepath = 'testdata.csv'
    data, input_scaler = load_and_preprocess_data(filepath)
    targets = prepare_targets(data, forecast_horizon=3)
    output_scaler = prepare_output_scaler(targets)

    x_train, x_val, y_train, y_val = train_test_split(data[:-3, :, :], targets, test_size=0.2, random_state=42)

    # Make sure to pass the correct input shape to the model creation function
    model = create_advanced_model((1, data.shape[2]))  # Adjust input_shape to (1, 8)
    trained_model = train_model(model, x_train, y_train, x_val, y_val)

    save_model_and_scaler(trained_model, input_scaler, 'smog_prediction_model.keras', 'input_scaler.gz')
    joblib.dump(output_scaler, 'output_scaler.gz')

    forecast = predict_and_format_output(trained_model, data[-3:, :, :], output_scaler)
    print(forecast)

if __name__ == "__main__":
    main()



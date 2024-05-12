import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import joblib

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['RollingMean_PM2.5'] = data['PM2.5'].rolling(window=3).mean().fillna(method='bfill')
    features = ['RollingMean_PM2.5', 'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[features])
    data_scaled = np.reshape(data_scaled, (data_scaled.shape[0], 1, data_scaled.shape[1]))
    return data_scaled, scaler

def create_advanced_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(9)  # Outputs for 3 days, 3 features each day
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=64)
    return model

def prepare_targets(data, forecast_horizon=3):
    targets = np.zeros((data.shape[0] - forecast_horizon, forecast_horizon * 3))
    for i in range(forecast_horizon):
        targets[:, i*3:(i+1)*3] = data[i+1:data.shape[0]-forecast_horizon+i+1, 0, [0, 4, 3]]  # PM2.5, WindDirection, WindSpeed
    return targets

def prepare_output_scaler(targets):
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    output_scaler.fit(targets)
    return output_scaler

def predict_and_format_output(model, data, output_scaler):
    # Predict using the model
    prediction = model.predict(data)

    # Ensure the prediction is reshaped to match the output scaler's expected input shape
    if prediction.shape[1] != 9:
        # This is a simple check and reshape; adjust as needed based on your model's specifics
        prediction = prediction.reshape(1, 9)

    # Inverse transform using the output scaler
    inversed_prediction = output_scaler.inverse_transform(prediction)

    # Formatting the output as JSON
    result = {}
    labels = ["smog_percentage", "wind_direction", "wind_speed"]
    for i in range(3):
        day_index = i + 1
        result[f"day_{day_index}"] = {
            labels[j]: float(inversed_prediction[0, i*3 + j]) for j in range(3)
        }
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

    model = create_advanced_model((1, 5))
    trained_model = train_model(model, x_train, y_train, x_val, y_val)

    save_model_and_scaler(trained_model, input_scaler, 'smog_prediction_model.keras', 'input_scaler.gz')
    joblib.dump(output_scaler, 'output_scaler.gz')

    forecast = predict_and_format_output(trained_model, data[-3:, :, :], output_scaler)
    print(forecast)

if __name__ == "__main__":
    main()

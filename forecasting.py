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
    # Reshape data to include timesteps (assuming 1 timestep for simplicity here)
    data_scaled = np.reshape(data_scaled, (data_scaled.shape[0], 1, data_scaled.shape[1]))
    return data_scaled, scaler

def create_advanced_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, x_val, y_val):
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=64)
    return model  # Return the model after it has been trained

def predict_and_format_output(model, data, scaler):
    prediction = model.predict(data)
    # Prepare a full feature set array with zeros or dummy values for all features except the target for inverse scaling
    full_features = np.zeros((prediction.shape[0], 5))  # Assuming 5 features including the target
    full_features[:, 0] = prediction.ravel()  # Assuming 'smog_percentage' is the first feature
    full_features = scaler.inverse_transform(full_features)
    smog_percentage = full_features[:, 0]  # Extract the inversely transformed target

    last_sequence = data[-1, -1, :]
    wind_direction = last_sequence[4]  # Assuming wind direction is the last feature
    wind_speed = last_sequence[3]  # Assuming wind speed is the second last feature

    result = {
        "smog_percentage": float(smog_percentage[0]),
        "wind_direction": float(wind_direction),
        "wind_speed": float(wind_speed)
    }
    return json.dumps(result)

def load_and_predict(input_data):
    model = load_model('smog_prediction_model.h5')
    scaler = joblib.load('smog_scaler.gz')
    prediction = model.predict(input_data)
    
    # Create a full-sized features array filled with zeros (or any nominal values) for inverse transformation
    full_prediction = np.zeros((prediction.shape[0], 5))  # Assuming 5 was the number of features the scaler was fit on
    full_prediction[:, 0] = prediction[:, 0]  # Assuming the prediction is the first feature

    # Apply inverse transformation
    inversed_prediction = scaler.inverse_transform(full_prediction)
    smog_percentage = inversed_prediction[:, 0]  # Extracting the smog percentage
    wind_direction = input_data[-1, -1, 4]  # Assuming static, example values; adapt as necessary
    wind_speed = input_data[-1, -1, 3]  # Assuming static, example values; adapt as necessary

    result = {
        "smog_percentage": float(smog_percentage[0]),
        "wind_direction": float(wind_direction),
        "wind_speed": float(wind_speed)
    }
    return json.dumps(result)


# Save the trained model and scaler to disk
def save_model_and_scaler(model, scaler):
    model.save('smog_prediction_model.h5')  # Save the model
    joblib.dump(scaler, 'smog_scaler.gz')  # Save the scaler

def main():
    filepath = 'testdata.csv'
    data, scaler = load_and_preprocess_data(filepath)
    print('Data shape after preprocessing:', data.shape)  # Should be (300, 1, 5)
    input_shape = (data.shape[1], data.shape[2])  # Should be (1, 5)
    print('Input shape:', input_shape)

    # Ensure all features are included in x and only the target variable is in y
    x_train, x_val, y_train, y_val = train_test_split(data[:, :, :], data[:, 0, 0], test_size=0.2, random_state=42)
    print('x_train shape:', x_train.shape)  # Should be (240, 1, 5)
    print('y_train shape:', y_train.shape)  # Should be (240,)

    model = create_advanced_model(input_shape)
    trained_model = train_model(model, x_train, y_train, x_val, y_val)  # trained_model is now the actual model
    save_model_and_scaler(trained_model, scaler)  # Save the actual model and scaler
    # Example prediction (adjust as needed)
    forecast = predict_and_format_output(trained_model, data[-1:, :, :], scaler)
    print(forecast)

if __name__ == "__main__":
    main()
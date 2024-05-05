import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(data, n_steps, future_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        out_end_ix = end_ix + future_steps
        if out_end_ix > len(data):
            break
        seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def preprocess_data(data):
    data = data[['PM2.5', 'PM10', 'VOCS', 'CO', 'O3']]
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    joblib.dump(scaler, 'scaler.gz')
    return data, scaler

def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

def smog_occurrence_likelihood(predictions):
    thresholds = {'PM2.5': 35, 'PM10': 50, 'VOCS': 0.6, 'CO': 9, 'O3': 70}
    exceedances = (predictions > np.array(list(thresholds.values()))).sum(axis=1)
    likelihood = exceedances / len(thresholds) * 100
    return likelihood

def train_and_predict(data, future_steps=3):
    data, scaler = preprocess_data(data)
    n_steps = 24
    X, y = prepare_data(data, n_steps, future_steps)
    model = build_model((n_steps, X.shape[2]), future_steps * data.shape[1])
    model.fit(X, y.reshape(y.shape[0], -1), epochs=50, batch_size=32, verbose=1)
    predictions = model.predict(X[-1].reshape(1, n_steps, X.shape[2]))
    predictions = predictions.reshape(future_steps, data.shape[1])
    model.save('my_model.keras')

    predictions_original_scale = scaler.inverse_transform(predictions)
    smog_likelihood = smog_occurrence_likelihood(predictions_original_scale)
    predictions_with_likelihood = np.hstack((predictions_original_scale, smog_likelihood[:, np.newaxis]))

    return predictions_with_likelihood

if __name__ == '__main__':
    data = pd.read_csv('testdata.csv')
    predictions = train_and_predict(data, future_steps=3)
    print("Predictions for the next three days with smog occurrence likelihood:", predictions)

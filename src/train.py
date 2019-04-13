import h5py
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, GRU
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adam, Adadelta
from keras.callbacks import ReduceLROnPlateau
from keras_contrib.callbacks import CyclicLR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
import keras
import re
import yaml
import gc
from sklearn.externals import joblib
from preprocess_data import production, consumption

HISTORY = 120
FORECAST = 1
VERBOSE = 1
EPOCHS = 100
BATCH_SIZE = 1024
RESAMPLE_FREQ = 20

def split_data(df, ratio=0.7):
    X, y = df[:int(len(df)*ratio)], df[int(len(df)*ratio):]
    return pd.DataFrame(X), pd.DataFrame(y)

def scale_data(train, test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return pd.DataFrame(train), pd.DataFrame(test)

def get_in_out_tgtidx(in_start):
    in_end = in_start + 120
    out_end = in_end + 1
    return in_end, out_end, 0

# convert history into inputs and outputs
def to_supervised(data_arr):
    X, y, in_start = list(), list(), 0
    for _ in range(len(data_arr)):
        in_end, out_end, target_index = get_in_out_tgtidx(in_start)
        if out_end < len(data_arr):  X.append(data_arr[in_start:in_end, :]); y.append(data_arr[in_end:out_end, target_index]); in_start += 1
    return np.array(X), np.array(y)

def drop_nans_from_data(X, y):
    train_cols = X.columns
    target_cols = y.columns

    temp_df = pd.merge(X, y, left_index=True, right_index=True)
    temp_df.dropna(how='any', inplace=True)

    return temp_df[train_cols], temp_df[target_cols]

def convert_to_df(data_frame):
    X, y = to_supervised(data_frame.values)
    X_cols = [str(col)+f'(t-{i})' for i in range(120-1, -1, -1) for col in data_frame.columns]
    y_cols = [f'true_future(t+{i+1})' for i in range(1)]
    return pd.DataFrame(X.reshape(-1, len(data_frame.columns)*120), columns=X_cols), pd.DataFrame(y, columns=y_cols)


def build_model(n_timesteps, n_outputs):
    model = Sequential(); model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, 1)))
    model.add(RepeatVector(n_outputs)); model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu'))); model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    return model

def reshape_data(X, y):
    X = X.values.reshape(-1, HISTORY, 1)
    y = np.expand_dims(y, axis=2)
    return X, y

if __name__ == "__main__":
    prefix = 'consumption'
    train, test = split_data(consumption)
    train, test = scale_data(train, test, prefix)
    X_train, y_train = convert_to_df(train, history=HISTORY, forecast=FORECAST)
    X_train, y_train = drop_nans_from_data(X_train, y_train)

    X_test, y_test = convert_to_df(test, history=HISTORY, forecast=FORECAST)
    X_test, y_test = drop_nans_from_data(X_test, y_test)
    X_train, y_train = reshape_data(X_train, y_train)
    X_test, y_test = reshape.data(X_test, y_test)
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = build_model(n_timesteps, n_features, n_outputs)
    model.compile(loss='mse', optimizer=Adam(lr=0.001, decay=1e-6, clipnorm=1.))

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.00001)

    # checkpoint
    filepath = f"{prefix}_-in{HISTORY}-out{FORECAST}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=VERBOSE, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, reduce_lr]

    # fit network
    LSTM_history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_data=(X_test, y_test), verbose=VERBOSE, shuffle=False,
                             callbacks=callbacks_list)


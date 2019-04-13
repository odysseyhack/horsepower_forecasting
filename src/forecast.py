import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from preprocess_data import production, consumption
from train import to_supervised, drop_nans_from_data, convert_to_df, split_data


HISTORY = 120
FORECAST = 1
VERBOSE = 1
EPOCHS = 20
BATCH_SIZE = 1024
LEARN_RATE = 0.001


def load_scaler(scaler_path):
    return joblib.load(scaler_path)


def scale_data(scaler, df):
    valid = df[: len(df)//2]
    return scaler.transform(valid.values.reshape(-1, 1))


def get_right_shape(df):
    X, y = convert_to_df(pd.DataFrame(df))
    X, y = drop_nans_from_data(X, y)
    return X.values.reshape(-1, 120, 1), np.expand_dims(y, axis=2)


def predict_multi(X, model):
    y_pred = model.predict(X)
    return y_pred.reshape(-1, 1)


def inverse_scaling(arr, scaler):
    arr = arr.reshape(-1, 1)
    return scaler.inverse_transform(arr)


if __name__=='__main__':
    production_scaler = load_scaler('../models/production_scaler.pkl')
    consumption_scaler = load_scaler('../models/production_scaler.pkl')

    production_model = load_model('../models/production_-in120-out1.hdf5')
    consumption_model = load_model('../models/consumption_-in120-out1.hdf5')

    production = scale_data(production_scaler, production)
    consumption = scale_data(consumption_scaler, consumption)

    prod_X, prod_y = get_right_shape(production)
    cons_X, cons_y = get_right_shape(consumption)

    prod_y_pred = predict_multi(prod_X, production_model)
    cons_y_pred = predict_multi(cons_X, consumption_model)

    inv_prod_y_pred = inverse_scaling(prod_y_pred, production_scaler)
    inv_prod_y = inverse_scaling(prod_y, production_scaler)

    inv_cons_y_pred = inverse_scaling(cons_y_pred, consumption_scaler)
    inv_cons_y = inverse_scaling(cons_y, consumption_scaler)

    print("Mean Absolute Error (production) :", mean_absolute_error(inv_prod_y, inv_prod_y_pred))
    print("Mean Absolute Error (consumption) :", mean_absolute_error(inv_cons_y, inv_cons_y_pred))

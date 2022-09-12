import socket
import pickle
import logging
import numpy as np

from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression

from collections import defaultdict
from config import Settings, config as hcnf

from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

import tensorflow_privacy
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models

dpg = hcnf.nn_config.dpg

cnf = Settings()
IP = socket.gethostbyname(cnf.IP)
PORT = cnf.SOCKETPORT
ADDR = (IP, PORT)
SIZE = cnf.SIZE

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression, features):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = features  # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def clientSocket(features):
    # Communicate features to the server.
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)
    logging.info(f"[CONNECTED] Client connected to server at {IP}:{PORT}")

    variable = features
    msg = pickle.dumps(variable)

    client.send(msg)

    variable = client.recv(SIZE)
    msg = pickle.loads(variable)
    client.close()

    return msg

def handle_client(conn, addr, var):
    logging.info(f"[NEW CONNECTION] {addr} connected.")

    # Recieve data from client
    data = conn.recv(SIZE)
    response = pickle.loads(data)
    # Logic
    if var == None:
        msg = pickle.dumps(response)
    else:
        msg = pickle.dumps(var)

    # Send and close
    conn.send(msg)
    conn.close()

    return response

def drop_columns(df_train):
    many_null_cols = [col for col in df_train.columns if df_train[col].isnull().sum() / df_train.shape[0] > 0.9]
    cols_to_drop = list(set(many_null_cols))
    return df_train.drop(cols_to_drop, axis=1)

def sample_data(X,t):
    "Resamples data: SMOTE and Random"
    k = 3
    over = SMOTE(sampling_strategy=0.4, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=0.5)
    pipeline = make_pipeline(over, under)

    X, t = pipeline.fit_resample(X, t)
    return X, t

def custom_cross(outlier_index, normal_index, folds):
    custom_cv = []

    for i in range(folds):
        splits = np.array_split(normal_index, folds)
        norm_split = splits[i]

        splits.pop(i)
        train_splits = np.concatenate(splits, axis=0)

        # Sample normal data based in outlier size
        test_index = np.random.choice(list(range(len(norm_split))), size = len(outlier_index))
        test_norm = norm_split[test_index]

        # Delete data from train normal data
        train_norm = np.delete(norm_split, test_index)

        # Save in data in custom cv.
        test = np.append(test_norm, outlier_index)
        train = np.append(train_norm, train_splits)
        custom_cv.append((train, test))

    return custom_cv

def evaluate_model(y_true, y_pred, y_prob):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    return [metrics.precision_score(y_true, y_pred, average = 'macro'), metrics.recall_score(y_true, y_pred, average = 'macro'), metrics.f1_score(y_true, y_pred, average='macro'), metrics.accuracy_score(y_true, y_pred), metrics.auc(fpr, tpr)]

def keras_cv(model, parameters, metrics_labels, split, X, t, epochs, batch_size):

    nn_scores = defaultdict(list)
    
    # Train with cross validation split
    for i, (train_index, test_index) in enumerate(split):

        # Initialize parameters
        model.set_weights(parameters)

        # Split data
        X_train, t_train = X.iloc[train_index], t.iloc[train_index]
        X_test, t_test = X.iloc[test_index], t.iloc[test_index]

        X_train, t_train = sample_data(X_train, t_train)

        # Train
        model.fit(X_train, t_train, epochs=epochs, batch_size=batch_size)

        # Save metrics
        y_pred_prob_2 = model.predict(X_test)
        y_pred_2 = y_pred_prob_2.round()
        results = evaluate_model(t_test, y_pred_2, y_pred_prob_2)

        for count, key in enumerate(metrics_labels):
            nn_scores[key].append(results[count])
    
    return nn_scores

def keras_cvJetson(model, parameters, metrics_labels, split, X, t, epochs, batch_size):

    nn_scores = defaultdict(list)
    
    # Train with cross validation split
    for i, (train_index, test_index) in enumerate(split):

        # Initialize parameters
        model.set_weights(parameters)

        # Split data
        X_train, t_train = X.iloc[train_index], t.iloc[train_index]
        X_test, t_test = X.iloc[test_index], t.iloc[test_index]

        X_train, t_train = sample_dataJetson(X_train, t_train)

        # Train
        model.fit(X_train, t_train, epochs=epochs, batch_size=batch_size)

        # Save metrics
        y_pred_prob_2 = model.predict(X_test)
        y_pred_2 = y_pred_prob_2.round()
        results = evaluate_model(t_test, y_pred_2)

        for count, key in enumerate(metrics_labels):
            nn_scores[key].append(results[count])
    
    return nn_scores

def autoKerasCV(model, parameters, metrics_labels, split, X, t, epochs, batch_size, norm_index):

    nn_scores = defaultdict(list)

    # Train with cross validation split
    for i, (train_index, test_index) in enumerate(split):
        # Initialize parameters
        model.set_weights(parameters)

        # Split data
        X_train, t_train = X.iloc[train_index], t.iloc[train_index]
        X_test, t_test = X.iloc[test_index], t.iloc[test_index]

        # Entrenamineto
        model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=1)

        y_train = model.predict(X.iloc[norm_index])
        mse_train = np.mean(np.power(X.iloc[norm_index] - y_train, 2), axis=1)
        umbral = np.max(mse_train)/10

        # Validation
        e_test = model.predict(X_test)
        mse_test = np.mean(np.power(X_test - e_test, 2), axis=1)

        y_pred = np.ones((t_test.shape))
        y_pred[mse_test > umbral] = -1
        results = evaluate_model(t_test, y_pred, y_pred)

        for count, key in enumerate(metrics_labels):
            nn_scores[key].append(results[count])
    
    return nn_scores

def hyper_nn(n_inputs, metrics, layer_sizes, learning_rate):
    # Define the model
    model = models.Sequential()
    model.add(keras.Input(shape=(n_inputs,)))
    
    for layer_size in layer_sizes:
        model.add(layers.Dense(layer_size, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
    model.add(layers.Dense(1, activation='sigmoid'))

    if dpg:
        noise_multiplier = 5
        l2_norm_clip = 5

        opt = tensorflow_privacy.DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=1,
        learning_rate=learning_rate)
        loss = keras.losses.BinaryCrossentropy(
        reduction = keras.losses.Reduction.NONE)
        # reduction is set to NONE to get loss in a vector form

        model.compile(loss=loss, optimizer=opt, metrics=metrics)
    else:
        # Define optimizer
        opt = keras.optimizers.Adam(learning_rate=learning_rate) 
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)
    
    return model

def hyper_auto(n_inputs, metrics, encoder, decoder, learning_rate):
    # Definimos primero las capas de la red
    model = models.Sequential()
    model.add(keras.Input(shape=(n_inputs,)))

    # Encoder
    for layer_size in encoder:
        model.add(layers.Dense(layer_size, activation="relu"))

    # Decoder
    for layer_size in decoder:
        model.add(layers.Dense(layer_size, activation="relu"))

    # Define output
    model.add(layers.Dense(n_inputs, activation='linear'))
    
    if dpg:
        noise_multiplier = 5
        l2_norm_clip = 5

        opt = tensorflow_privacy.DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=1,
        learning_rate=learning_rate)
        loss = keras.losses.BinaryCrossentropy(
        reduction = keras.losses.Reduction.NONE)
        # reduction is set to NONE to get loss in a vector form

        model.compile(loss=loss, optimizer=opt, metrics=metrics)
    else:
        # Posteriormente, especificamos el optimizador a emplear y sus hiperparámetros
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        # optimizador, función de error, métricas de evaluación, etc.
        model.compile(loss='mse', optimizer=opt, metrics=metrics)

    return model

def sample_dataJetson(X,t):
    "Resamples data: SMOTE and Random"
    k = 3
    over = SMOTE(ratio=0.4, k_neighbors=k)
    under = RandomUnderSampler(ratio=0.5)
    pipeline = make_pipeline(over, under)

    X, t = pipeline.fit_sample(X, t)
    return X, t

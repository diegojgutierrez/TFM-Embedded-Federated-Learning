import os
import sys
import time
import socket
import pickle
import logging
import datetime
import flwr as fl
import numpy as np
import pandas as pd

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import utils
from config import Settings

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

metrics_labels = ['test_precision_macro', 'test_recall_macro', 'test_f1_macro', 'test_accuracy','test_roc_auc']

cnf = Settings()
IP = socket.gethostbyname(cnf.IP)
PORT = cnf.SOCKETPORT
ADDR = (IP, PORT)
SIZE = cnf.SIZE

class AutoClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self):
        raise Exception("Not implemented (server-side parameter initialization)")
    
    def fit(self, parameters, config):

        # Receive hyperparameters
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]

        encoder = list(map( int, config["layers"].split(',')))
        decoder = encoder[::-1]
        decoder.pop(0)
        # Configure model
        if self.model == None:
            self.model = utils.hyper_auto(m, ['acc'], encoder, decoder, config["learning_rate"])

        # Intialize NN    
        K.clear_session()
        self.model.set_weights(parameters)

        scores = utils.autoKerasCV(self.model, parameters, metrics_labels, custom_cv, X_cv, y_cv, epochs, batch_size, norm_index)

        self.model.set_weights(parameters)
        self.model.fit(X_cv.iloc[norm_index], X_cv.iloc[norm_index], epochs=epochs, batch_size=batch_size, verbose=1)

        y_train = self.model.predict(X_cv.iloc[norm_index])
        mse_train = np.mean(np.power(X_cv.iloc[norm_index] - y_train, 2), axis=1)
        self.umbral = np.max(mse_train)/10

        # Validación
        e_test = self.model.predict(X_res)
        mse_test = np.mean(np.power(X_res - e_test, 2), axis=1)
        loss = mean_squared_error(X_res, e_test)

        y_pred = np.ones((y_res.shape))
        y_pred[mse_test > self.umbral] = -1
        results = utils.evaluate_model(y_res, y_pred, y_pred)

        nn_scores = {}
        for count, key in enumerate(metrics_labels):
            nn_scores[key] = results[count]

        logging.info("Evaluration pre merge done")

        # Fill dictionary with results
        results = {"cv_accuracy": np.mean(scores['test_accuracy']), 
        "cv_roc_auc": np.mean(scores['test_roc_auc']), 
        "cv_Precision": np.mean(scores['test_precision_macro']), 
        "cv_f1": np.mean(scores['test_f1_macro']), 
        "cv_recall": np.mean(scores['test_recall_macro']),
        "pre_accuracy": nn_scores['test_accuracy'], 
        "pre_roc_auc": nn_scores['test_roc_auc'], 
        "pre_Precision": nn_scores['test_precision_macro'], 
        "pre_f1": nn_scores['test_f1_macro'], 
        "pre_recall": nn_scores['test_recall_macro'],
        "pre_num": len(X_res),
        "pre_loss": loss,
        "treshold": self.umbral}

        return self.model.get_weights(), len(X_cv.iloc[norm_index]), results

    def evaluate(self, parameters, config):
        encoder = list(map( int, config["layers"].split(',')))
        decoder = encoder[::-1]
        decoder.pop(0)
        self.umbral = config["umbral"]

        # Configure model
        if self.model == None:
            self.model = utils.hyper_auto(m, ['acc'], encoder, decoder, config["learning_rate"])
        K.clear_session()
        self.model.set_weights(parameters)

        # Validación
        e_test = self.model.predict(X_res)
        mse_test = np.mean(np.power(X_res - e_test, 2), axis=1)
        loss = mean_squared_error(X_res, e_test)

        y_pred = np.ones((y_res.shape))
        y_pred[mse_test > self.umbral] = -1
        results = utils.evaluate_model(y_res, y_pred, y_pred)

        fpr, tpr, thresholds = metrics.roc_curve(y_res, y_pred)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % metrics.auc(fpr, tpr))
        plt.legend(loc = 'lower right')
        plt.plot([-0.02, 1.02], [-0.02, 1.02],'r--')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("../../reports/figures/autoencoder/test_" + str(sys.argv[1]) + "_" + str(datetime.datetime.now()) + ".png")
        plt.clf()

        cprecision, crecall, thresholds = metrics.precision_recall_curve(y_res, y_pred)

        plt.title('Precision Recall Curve')
        plt.plot(crecall, cprecision, 'b', label = 'AP = %0.2f' % metrics.average_precision_score(y_res, y_pred))
        plt.legend(loc = 'lower right')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig("../../reports/figures/autoencoder/testpr_" + str(sys.argv[1]) + "_" + str(datetime.datetime.now()) + ".png")
        plt.clf()
        logging.info("Evaluration post merge done")

        nn_scores = {}
        for count, key in enumerate(metrics_labels):
            nn_scores[key] = results[count]

        return loss, len(X_res), {"post_accuracy": nn_scores['test_accuracy'], "post_roc_auc": nn_scores['test_roc_auc'], "post_Precision": nn_scores['test_precision_macro'], "post_f1": nn_scores['test_f1_macro'], "post_recall": nn_scores['test_recall_macro'], "post_loss":loss}

def main():
    global X_cv, y_cv, X_res, y_res, m, norm_index, custom_cv

    train = pd.read_csv("../../data/processed/train_data_" + sys.argv[1] + ".csv")
    initial_run = sys.argv[2].lower() == 'true'

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("../logs/client_" + str(sys.argv[1]) + ".log"),
        logging.StreamHandler()
    ]
    )

    train = utils.drop_columns(train)
    train['Amount'] = boxcox1p(train['Amount'], boxcox_normmax(train['Amount'] + 1))

    # Reduce memory usage of dataset
    for df in [train]:
        original = df.copy()
        df = utils.reduce_mem_usage(df)

        for col in list(df):
            if df[col].dtype!='O':
                if (df[col]-original[col]).sum()!=0:
                    df[col] = original[col]

    # Process data
    X = train.drop("Class", 1)
    Y = train["Class"]
    Y = Y.replace({0:1, 1:-1})

    # Get features
    features = X.columns.values.tolist()

    if initial_run:
        # Intersection of features 
        utils.clientSocket(features)
        time.sleep(20)
        features = utils.clientSocket(features)
        time.sleep(10)
        with open('outfile', 'wb') as fp:
            pickle.dump(features, fp)
    else:
        with open ('outfile', 'rb') as fp:
            features = pickle.load(fp)
            
    X = X[features]

    test_size = 0.25
    X_cv, X_res, y_cv, y_res = train_test_split(X, Y, stratify=Y, test_size=test_size)

    _, m = X_cv.shape

    # Scale data
    scaler = StandardScaler()
    X_cv[:] = scaler.fit_transform(X_cv.values)
    X_res[:] = scaler.transform(X_res.values)

    # Send outliers to testing
    norm_index = np.where(y_cv == 1)[0]
    outlier_index = np.where(y_cv == -1)[0]
    X_res = pd.concat([X_res, X_cv.iloc[outlier_index]])
    y_res = pd.concat([y_res, y_cv.iloc[outlier_index]])

    folds = 5
    custom_cv = utils.custom_cross(outlier_index, norm_index, folds)

    fl.client.start_numpy_client(IP + ":8080", client=AutoClient(None))

if __name__ == "__main__":
    main()
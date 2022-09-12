import os
import sys
import time
import pickle
import socket
import logging
import datetime
import flwr as fl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import backend as K

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import utils
from config import Settings

cnf = Settings()
IP = socket.gethostbyname(cnf.IP)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

nn_metrics = ['AUC']
metrics_labels = ['test_precision_macro', 'test_recall_macro', 'test_f1_macro', 'test_accuracy','test_roc_auc']

class NNClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self):
        #return model.get_weights()
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):

        # Receive hyperparameters
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]

        # Configure model
        if self.model == None:
            self.model = utils.hyper_nn(m, nn_metrics, list(map( int, config["layers"].split(','))), config["learning_rate"])
        K.clear_session()
        self.model.set_weights(parameters)

        # Cross validation
        k_folds = 5
        skf = StratifiedKFold(n_splits=k_folds)
        scores = utils.keras_cvJetson(self.model, parameters, metrics_labels, skf.split(X_cv, y_cv), X_cv, y_cv, epochs, batch_size)
        
        # Full train  model
        self.model.set_weights(parameters)
        self.model.fit(X_fcv, y_fcv, epochs=epochs, batch_size=batch_size, verbose=1)

        # Evalutation
        loss, _ = self.model.evaluate(X_res, y_res)
        y_pred_prob_2 = self.model.predict(X_res)
        y_pred_2 = y_pred_prob_2.round()

        # Calculate metrics
        precision, recall, f1_score, accuracy, auc = utils.evaluate_model(y_res, y_pred_2, y_pred_prob_2)
        logging.info("Evaluration pre merge done")

        # Fill dictionary with results
        results = {"cv_accuracy": np.mean(scores['test_accuracy']), 
        "cv_roc_auc": np.mean(scores['test_roc_auc']), 
        "cv_Precision": np.mean(scores['test_precision_macro']), 
        "cv_f1": np.mean(scores['test_f1_macro']), 
        "cv_recall": np.mean(scores['test_recall_macro']),
        "pre_accuracy": accuracy, 
        "pre_roc_auc": auc, 
        "pre_Precision": precision, 
        "pre_f1": f1_score, 
        "pre_recall": recall,
        "pre_num": len(X_res),
        "pre_loss":loss}

        model_parameters = self.model.get_weights()
        
        return model_parameters, len(X_cv), results

    def evaluate(self, parameters, config):
        if self.model == None:
            self.model = utils.hyper_nn(m, nn_metrics, list(map( int, config["layers"].split(','))), config["learning_rate"])

        self.model.set_weights(parameters)

        if cnf.SAVE:
            self.model.save('my_model.h5')
        
        # Evaluate
        loss, _ = self.model.evaluate(X_res, y_res)
        y_pred_prob_2 = self.model.predict(X_res)
        y_pred_2 = y_pred_prob_2.round()

        # Calculate metrics
        precision, recall, f1_score, accuracy, auc = utils.evaluate_model(y_res, y_pred_2, y_pred_prob_2)

        fpr, tpr, thresholds = metrics.roc_curve(y_res, y_pred_prob_2)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % metrics.auc(fpr, tpr))
        plt.legend(loc = 'lower right')
        plt.plot([-0.02, 1.02], [-0.02, 1.02],'r--')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("../../reports/figures/nn/test_" + str(sys.argv[1]) + "_" + str(datetime.datetime.now()) + ".png")
        plt.clf()

        cprecision, crecall, thresholds = metrics.precision_recall_curve(y_res, y_pred_prob_2)

        plt.title('Precision Recall Curve')
        plt.plot(crecall, cprecision, 'b', label = 'AP = %0.2f' % metrics.average_precision_score(y_res, y_pred_2))
        plt.legend(loc = 'lower right')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig("../../reports/figures/nn/testpr_" + str(sys.argv[1]) + "_" + str(datetime.datetime.now()) + ".png")
        plt.clf()
        logging.info("Evaluration post merge done")

        return loss, len(X_res), {"post_accuracy": accuracy, "post_Precision": precision, "post_f1": f1_score, "post_roc_auc": auc, "post_recall":recall, "post_loss":loss}

def main():
    global X_cv, X_fcv, X_res, y_cv, y_res, y_fcv, m

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
                    print('Bad transformation', col)

    X = train.drop("Class", 1)
    Y = train["Class"]

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

    # Split data
    test_size = 0.25
    X_cv, X_res, y_cv, y_res = train_test_split(X, Y, stratify=Y, test_size=test_size)

    # Resample data
    X_fcv, y_fcv = utils.sample_dataJetson(X_cv, y_cv)

    X_fcv = pd.DataFrame(X, columns = features)
    y_fcv = pd.Series(Y)

    # Scale data
    scaler = StandardScaler()
    X_cv[:] = scaler.fit_transform(X_cv.values)
    X_fcv[:] = scaler.fit_transform(X_fcv.values)
    X_res[:] = scaler.transform(X_res.values)

    _, m = X.shape
    del Y, train

    fl.client.start_numpy_client(IP + ":8080", client=NNClient(None))

if __name__ == "__main__":
    main()
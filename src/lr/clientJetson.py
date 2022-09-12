import os
import sys
import time
import logging
import datetime
import flwr as fl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import socket
import pickle
import warnings

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, StratifiedKFold

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import utils
from config import Settings

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

cnf = Settings()
IP = socket.gethostbyname(cnf.IP)
PORT = cnf.SOCKETPORT
ADDR = (IP, PORT)
SIZE = cnf.SIZE

class SKClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self):
        #return model.get_weights()
        raise Exception("Not implemented (server-side parameter initialization)")
        #return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):
        # Receive hyperparameters

        # Con regresión logística
        self.model = LogisticRegression(penalty=config["penalty"], solver = config["solver"], max_iter=config["local_epochs"], C=config["C"], warm_start=True)
        utils.set_initial_params(self.model, m)
        self.model = utils.set_model_params(self.model, parameters)

        k = 3
        over = SMOTE(ratio=0.4, k_neighbors=k)
        under = RandomUnderSampler(ratio=0.5)
        pipeline = make_pipeline(over, under)

        # Cross Validate
        k_folds = 5
        skf = StratifiedKFold(n_splits=k_folds)
        scoring = ['precision_macro', 'recall_macro', 'f1_macro','accuracy', 'roc_auc']
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_validate(pipeline, X_cv, y_cv, cv= skf.split(X_cv,y_cv), scoring=scoring)

        # Reinitialize model
        self.model = LogisticRegression(penalty=config["penalty"], solver = config["solver"], max_iter=config["local_epochs"], C=config["C"], warm_start=True)
        utils.set_initial_params(self.model, m)
        self.model = utils.set_model_params(self.model, parameters)
        pipeline = make_pipeline(over, under, self.model)

        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_cv, y_cv)
        y_pred = pipeline.predict(X_res)
        y_pred_prob = self.model.predict_proba(X_res)[::,1]
        eval_scores = utils.evaluate_model(y_res, y_pred, y_pred_prob)
        loss = log_loss(y_res, self.model.predict_proba(X_res))
        logging.info("Evaluration pre merge done")

        results = {"cv_accuracy": np.mean(scores['test_accuracy']), 
        "cv_roc_auc": np.mean(scores['test_roc_auc']), 
        "cv_Precision": np.mean(scores['test_precision_macro']), 
        "cv_f1": np.mean(scores['test_f1_macro']), 
        "cv_recall": np.mean(scores['test_recall_macro']),
        "pre_accuracy": eval_scores[3], 
        "pre_roc_auc": eval_scores[4],
        "pre_Precision": eval_scores[0],
        "pre_f1": eval_scores[2], 
        "pre_recall": eval_scores[1],
        "pre_num": len(X_res),
        "pre_loss":loss}

        self.model = pipeline[2]
        model_parameters = utils.get_model_parameters(self.model)
        return model_parameters, len(X_cv), results

    def evaluate(self, parameters, config):
        self.model = LogisticRegression(penalty=config["penalty"], solver = config["solver"], max_iter=config["local_epochs"], C=config["C"], warm_start=True)
        utils.set_initial_params(self.model, m)
        self.model = utils.set_model_params(self.model, parameters)

        y_pred = self.model.predict(X_res)
        y_pred_prob = self.model.predict_proba(X_res)[::,1]
        eval_scores = utils.evaluate_model(y_res, y_pred, y_pred_prob)
        loss = log_loss(y_res, y_pred_prob)

        fpr, tpr, thresholds = metrics.roc_curve(y_res, y_pred_prob)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % metrics.auc(fpr, tpr))
        plt.legend(loc = 'lower right')
        plt.plot([-0.02, 1.02], [-0.02, 1.02],'r--')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("../../reports/figures/lr/test_" + str(sys.argv[1]) + "_" + str(datetime.datetime.now()) + ".png")
        plt.clf()

        cprecision, crecall, thresholds = metrics.precision_recall_curve(y_res, y_pred_prob)

        plt.title('Precision Recall Curve')
        plt.plot(crecall, cprecision, 'b', label = 'AP = %0.2f' % metrics.average_precision_score(y_res, y_pred))
        plt.legend(loc = 'lower right')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig("../../reports/figures/lr/testpr_" + str(sys.argv[1]) + "_" + str(datetime.datetime.now()) + ".png")
        plt.clf()
        logging.info("Evaluration post merge done")
        return loss, len(X_res), {"post_accuracy": eval_scores[3], "post_Precision": eval_scores[0], "post_f1": eval_scores[2], "post_roc_auc": eval_scores[4], "post_recall":eval_scores[1], "post_loss":loss}


def main():
    global X_cv, X_res, y_cv, y_res, m

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

    test_size = 0.25
    X_cv, X_res, y_cv, y_res = train_test_split(X, Y, stratify=Y, test_size=test_size)

    # Scale data
    scaler = StandardScaler()
    X_cv[:] = scaler.fit_transform(X_cv.values)
    X_res[:] = scaler.transform(X_res.values)

    n, m = X.shape
    del Y, train

    fl.client.start_numpy_client(IP + ":8080", client=SKClient(None))

if __name__ == "__main__":
    main()
import os
import sys
import socket
import logging
import flwr as fl
import pandas as pd
import concurrent.futures

from sklearn.linear_model import LogisticRegression

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import utils
from config import Settings, config as hcnf
from strategies import CustomStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("../logs/server.log"),
        logging.StreamHandler()
    ]
)

cnf = Settings()

IP = socket.gethostbyname(cnf.IP)
PORT = cnf.SOCKETPORT
ADDR = (IP, PORT)
SIZE = cnf.SIZE

rounds = cnf.ROUNDS
fit_clients = cnf.FIT_CLIENTS
eval_clients = cnf.EVAL_CLIENTS
num_client = cnf.NUM_CLIENTS

# Logistic Regression
solver = hcnf.lg_config.solver
penalty = hcnf.lg_config.penalty
C = hcnf.lg_config.C
max_iter = hcnf.lg_config.max_iter
modelName = hcnf.lg_config.name
dpg = hcnf.nn_config.dpg


def fit_config(rnd: int):
    return strategy.config

def evaluate_config(rnd: int):
    return strategy.config

def main():
    global strategy

    # Socket server for variable intersection
    logging.info("[STARTING] Server is starting...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    logging.info(f"[LISTENING] Server is listening on {IP}:{PORT}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_client) as executor:

        results = []
        futures = []
        for cn in range(num_client):
            conn, addr = server.accept()
            futures.append(executor.submit(utils.handle_client, conn, addr, None))
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

        var = list(set(results[0]).intersection(*results))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_client) as executor:

        results = []
        futures = []
        for cn in range(num_client):
            conn, addr = server.accept()
            futures.append(executor.submit(utils.handle_client, conn, addr, var))

    i = 0
    for s in solver:
        for p in penalty:
            for cs in C:
                for local_epochs in max_iter:
                    logging.info("=============== Iteration: " + str(i) + " ===============")
                    logging.info("Configuración: ")
                    logging.info("Solver: " + str(s))
                    logging.info("Penalty: " + str(p))
                    logging.info("C: " + str(cs))
                    logging.info("Max iter: " + str(local_epochs))

                    # Change hyperparameters
                    config = {"solver": s, "penalty": p, "C": cs, "local_epochs": local_epochs}
                    config_df = pd.DataFrame(config, index = [0])
                    config_df.to_csv('../../reports/tables/params/' + str(modelName) +'_' + str(dpg) + '_' + 'dpg' + '_hyperparams_' + str(i) + '.csv', index=False)

                    # Define model
                    m = len(var)
                    model = LogisticRegression(penalty=config["penalty"], solver = config["solver"], max_iter=config["local_epochs"])
                    utils.set_initial_params(model, m)
                    model_parameters = fl.common.weights_to_parameters(utils.get_model_parameters(model))

                    # Federated training
                    strategy = CustomStrategy(fraction_fit = 0.25, fraction_eval=1, min_fit_clients=fit_clients, min_eval_clients= eval_clients, min_available_clients=num_client , on_fit_config_fn=fit_config, on_evaluate_config_fn=evaluate_config, initial_parameters=model_parameters, init_config=config, i=i, model_name=modelName)

                    # Start Flower server
                    fl.server.start_server(
                        server_address="[::]:8080",
                        config={"num_rounds": rounds},
                        strategy=strategy
                    )
                    i += 1

if __name__ == "__main__":
    main()
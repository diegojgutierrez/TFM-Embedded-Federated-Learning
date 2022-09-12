import os
import sys
import socket
import pickle
import logging
import flwr as fl
import pandas as pd
import concurrent.futures

import tensorflow as tf
from tensorflow.keras import backend as K

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

# Neural Net hyperparameters
learning_rates = hcnf.nn_config.learning_rates
batch_size = hcnf.nn_config.batch_size
epochs = hcnf.nn_config.epochs
layers = hcnf.nn_config.layers
modelName = hcnf.nn_config.name
dpg = hcnf.nn_config.dpg

MEMORY_LIMIT = 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])
    except RuntimeError as e:
        logging.error(e)

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
    for ly in layers:
        for batch in batch_size:
            for epoch in epochs:
                for lr in learning_rates:
                    
                    logging.info("=============== Iteration: " + str(i) + " ===============")
                    logging.info("Configuración: ")
                    logging.info("Layers: " + str(ly))
                    logging.info("Batch size: "+ str(batch))
                    logging.info("Epochs: "+ str(epoch))
                    logging.info("Learning rate: "+ str(lr))

                    # Change hyperparameters
                    config = {
                        "rounds": rounds,
                        "batch_size": batch,
                        "local_epochs": epoch,
                        "learning_rate": lr,
                        "optimizer": "adam",
                        "loss_fn": 'binary_crossentropy',
                        "layers" : ",".join([str(layer) for layer in ly]),
                        "umbral": 0
                    }

                    # Create hyperparam table
                    config_df = pd.DataFrame(config, index = [0])
                    config_df.to_csv('../../reports/tables/params/' + str(modelName) +'_' + str(dpg) + '_' + 'dpg' + '_hyperparams_' + str(i) + '.csv', index=False)

                    # Define model to send parameters
                    m = len(var)
                    model = utils.hyper_nn(m, ['AUC'], ly, lr)
                    model_parameters = fl.common.weights_to_parameters(model.get_weights())

                    # Clean model
                    del model
                    K.clear_session()

                    # Federated training
                    strategy = CustomStrategy(fraction_fit = 0.3, fraction_eval=1, min_fit_clients=fit_clients, min_eval_clients= eval_clients, min_available_clients=num_client , on_fit_config_fn=fit_config, on_evaluate_config_fn=evaluate_config, initial_parameters=model_parameters, init_config = config, i=i, model_name=modelName)

                    # Start Flower server
                    fl.server.start_server(
                        server_address="[::]:8080",
                        config={"num_rounds": rounds},
                        strategy=strategy
                    )
                    i += 1

if __name__ == "__main__":
    main()
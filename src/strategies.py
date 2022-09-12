import logging
import flwr as fl
import numpy as np
import pandas as pd

from config import Settings, config as hcnf
from collections import defaultdict
from typing import List, Optional, Tuple

cnf = Settings()

dpg = hcnf.nn_config.dpg

fit_clients = cnf.FIT_CLIENTS
eval_clients = cnf.EVAL_CLIENTS
num_client = cnf.NUM_CLIENTS

# Result tags
cv_metrics = ['cv_roc_auc', 'cv_accuracy', 'cv_Precision', 'cv_recall', 'cv_f1', 'examples']
pre_metrics = ['pre_roc_auc', 'pre_accuracy', 'pre_Precision', 'pre_recall', 'pre_f1', "pre_loss", "pre_num"]
post_metrics = ['post_roc_auc', 'post_accuracy', 'post_Precision', 'post_recall', 'post_f1', "post_loss", "examples"]
post_metrics_1 = ['post_roc_auc', 'post_accuracy', 'post_Precision', 'post_recall', 'post_f1', "post_loss","post_num"]

class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, fraction_fit, fraction_eval, min_fit_clients, min_eval_clients, min_available_clients, on_fit_config_fn, on_evaluate_config_fn, initial_parameters, init_config, i, model_name):
        self.config = init_config
        self.i = i
        self.steps = 0
        self.modelName = model_name
        super().__init__(fraction_fit=fraction_fit, fraction_eval=fraction_eval, min_fit_clients=min_fit_clients, min_eval_clients=min_eval_clients, min_available_clients=min_available_clients, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, initial_parameters=initial_parameters)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:

        global steps

        # Aggregate Cross Validation results
        cv_results = defaultdict(list)
        for metric in cv_metrics:
            weighted_metric = 0
            total_weights = 0
            sum_metric = 0
            for _, r in results:
                if metric != "examples":
                    cv_results[metric].append(r.metrics[metric])
                    weighted_metric += r.metrics[metric] * r.num_examples
                    total_weights += r.num_examples
                    sum_metric += r.metrics[metric]
                else:
                    cv_results[metric].append(r.num_examples)
            if metric != "examples":
                cv_results[metric].append(weighted_metric/total_weights)
                cv_results[metric].append(sum_metric/fit_clients)
            else:
                cv_results[metric].append(np.nan)
                cv_results[metric].append(np.nan)

        # Aggregate Pre-merge evaluation results
        for metric in pre_metrics:
            weighted_metric = 0
            total_weights = 0
            sum_metric = 0
            for _, r in results:
                if metric != "pre_num":
                    cv_results[metric].append(r.metrics[metric])
                    weighted_metric += r.metrics[metric] * r.metrics["pre_num"]
                    total_weights += r.metrics["pre_num"]
                    sum_metric += r.metrics[metric]
                else: 
                    cv_results[metric].append(r.metrics["pre_num"])

            if metric != "pre_num":
                cv_results[metric].append(weighted_metric/total_weights)
                cv_results[metric].append(sum_metric/fit_clients)
            else:
                cv_results[metric].append(np.nan)
                cv_results[metric].append(np.nan)

        cv_df = pd.DataFrame(cv_results, columns=cv_metrics + pre_metrics)
        cv_df.to_csv('../../reports/tables/results/'  + str(self.modelName) + '_' + str(dpg) + '_' +  'dpg' + '_results_train_iteration_' + str(self.i) + "_round_" + str(self.steps) + '.csv', index=False, header=cv_metrics + pre_metrics)

        if self.modelName == "Autoencoder":
            total_weights = 0
            sum_treshold = 0
            for _, r in results:
                sum_treshold += r.metrics["treshold"]* r.num_examples
                total_weights += r.num_examples
            self.config["umbral"] = sum_treshold/total_weights
            logging.info("Umbral agregado: " + str(self.config["umbral"]))
        
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            np.savez("../../models/" + self.modelName + "/round-" + str(rnd) + "-weights.npz", *aggregated_weights)
        return aggregated_weights

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""

        global steps

        cv_results = defaultdict(list)
        for metric in post_metrics: 
            weighted_metric = 0
            total_weights = 0
            sum_metric = 0
            for _, r in results:
                if metric != "examples":
                    cv_results[metric].append(r.metrics[metric])
                    weighted_metric += r.metrics[metric] * r.num_examples
                    total_weights += r.num_examples
                    sum_metric += r.metrics[metric]
                else:
                    cv_results["post_num"].append(r.num_examples)
            if metric != "examples":
                cv_results[metric].append(weighted_metric/total_weights)
                cv_results[metric].append(sum_metric/eval_clients)
            else:
                cv_results["post_num"].append(np.nan)
                cv_results["post_num"].append(np.nan)

        cv_df = pd.DataFrame(cv_results, columns=post_metrics_1)
        cv_df.to_csv('../../reports/tables/results/' + str(self.modelName) + '_' + str(dpg) + '_' + 'dpg' + '_results_evaluation_iteration_' + str(self.i) + "_round_" + str(self.steps) + '.csv', index=False, header=post_metrics_1)
        self.steps += 1

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)
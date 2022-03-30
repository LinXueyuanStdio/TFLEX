import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid

from temp.load_data import Data
from temp.train import train_temporal
from toolbox.nn.TuckERCPD import TuckERCPD
from toolbox.nn.TuckERTTR import TuckERTTR


def grid_search(model, data, param_model_grid, learning_grid, file, cols):
    # Get param list to try
    param_model_list = list(ParameterGrid(param_model_grid))
    learning_list = list(ParameterGrid(learning_grid))

    n_training = len(param_model_list) * len(learning_list)

    score_table = []

    for i_param_model in param_model_list:
        for i_learning_param in learning_list:
            print(f"{i_param_model}\n{i_learning_param}\n")

            t_start = time.time()

            # Create model
            modeli = model(d=data, **i_param_model)

            # Train 
            modeli, test_metrics = train_temporal(modeli, data, **i_learning_param)

            # Save data to file
            total_time = time.time() - t_start
            score_table.append([i_param_model, i_learning_param, *test_metrics, total_time])
            pd.DataFrame(score_table, columns=cols).to_csv(file)

            # Clear memory cache
            del modeli
            torch.cuda.empty_cache()

    return score_table


def main():
    # load data and preprocess
    data = Data()

    device = 'cuda'  # 'cpu' or 'cuda'

    # Parameter Grid to test
    parameter_grid_tr = {'de': np.linspace(50, 300, 3, dtype=int), 'dr': [40], 'dt': [40], 'ranks': [10], 'device': [device], "input_dropout": [0.], "hidden_dropout1": [0.], "hidden_dropout2": [0.]}
    parameter_grid_cpd = {'de': np.linspace(50, 300, 3, dtype=int), 'dr': [40], 'dt': [40], 'device': [device], "input_dropout": [0.], "hidden_dropout1": [0.], "hidden_dropout2": [0.]}

    learning_param_grid = {'learning_rate': [0.003, 0.002, 0.001, 0.0008, 0.0005, 0.0002, 0.0001], 'n_iter': [400], 'batch_size': [128], 'device': [device]}

    cols = ["Model Parameters", "Learning Parameters", "Test MRR", "Test Hits@1", "Test Hits@3", "Test Hits@10", "Time"]

    print("-------------------------------------------------------------------\n Training TuckER with CPD on core \n--------------------------------------------------------------------")
    scores_cpd = grid_search(TuckERCPD, data, parameter_grid_cpd, learning_param_grid, file="results/CPD.csv", cols=cols)

    print(
        "-------------------------------------------------------------------\n Traning TuckER with Tensor Ring core decomposition \n-------------------------------------------------------------------")
    scores_tr = grid_search(TuckERTTR, data, parameter_grid_tr, learning_param_grid, file='results/TR.csv', cols=cols)


if __name__ == "__main__":
    main()

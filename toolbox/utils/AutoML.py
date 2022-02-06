import numpy as np


def config_tuning_space(tuning_space_raw):
    from hyperopt import hp
    from hyperopt.pyll.base import scope
    if tuning_space_raw is None:
        return None

    hyper_obj = {}
    if "learning_rate" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{
            "learning_rate": hp.loguniform('learning_rate', np.log(tuning_space_raw['learning_rate']['min']),
                                           np.log(tuning_space_raw['learning_rate']['max']))}}
    if "hidden_size" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"hidden_size": scope.int(
            hp.qloguniform('hidden_size', np.log(tuning_space_raw['hidden_size']['min']),
                           np.log(tuning_space_raw['hidden_size']['max']), 1))}}
    if "ent_hidden_size" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"ent_hidden_size": scope.int(
            hp.qloguniform("ent_hidden_size", np.log(tuning_space_raw['ent_hidden_size']['min']),
                           np.log(tuning_space_raw['ent_hidden_size']['max']), 1))}}
    if "rel_hidden_size" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"rel_hidden_size": scope.int(
            hp.qloguniform("rel_hidden_size", np.log(tuning_space_raw['rel_hidden_size']['min']),
                           np.log(tuning_space_raw['rel_hidden_size']['max']), 1))}}
    if "batch_size" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"batch_size": scope.int(
            hp.qloguniform("batch_size", np.log(tuning_space_raw['batch_size']['min']),
                           np.log(tuning_space_raw['batch_size']['max']), 1))}}
    if "margin" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{
            "margin": hp.uniform("margin", tuning_space_raw["margin"]["min"], tuning_space_raw["margin"]["max"])}}
    if "lmbda" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"lmbda": hp.loguniform('lmbda', np.log(tuning_space_raw["lmbda"]["min"]),
                                                            np.log(tuning_space_raw["lmbda"]["max"]))}}
    if "distance_measure" in tuning_space_raw:
        hyper_obj = {**hyper_obj,
                     **{"distance_measure": hp.choice('distance_measure', tuning_space_raw["distance_measure"])}}
    if "cmax" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"cmax": hp.loguniform('cmax', np.log(tuning_space_raw["cmax"]["min"]),
                                                           np.log(tuning_space_raw["cmax"]["max"]))}}
    if "cmin" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"cmin": hp.loguniform('cmin', np.log(tuning_space_raw["cmin"]["min"]),
                                                           np.log(tuning_space_raw["cmin"]["max"]))}}
    if "optimizer" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"optimizer": hp.choice("optimizer", tuning_space_raw["optimizer"])}}
    if "bilinear" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"bilinear": hp.choice('bilinear', tuning_space_raw["bilinear"])}}
    if "epochs" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"epochs": hp.choice("epochs", tuning_space_raw["epochs"])}}
    if "feature_map_dropout" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{
            "feature_map_dropout": hp.choice('feature_map_dropout', tuning_space_raw["feature_map_dropout"])}}
    if "input_dropout" in tuning_space_raw:
        hyper_obj = {**hyper_obj,
                     **{"input_dropout": hp.choice('input_dropout', tuning_space_raw["input_dropout"])}}
    if "hidden_dropout" in tuning_space_raw:
        hyper_obj = {**hyper_obj,
                     **{"hidden_dropout": hp.choice('hidden_dropout', tuning_space_raw["hidden_dropout"])}}
    if "use_bias" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"use_bias": hp.choice('use_bias', tuning_space_raw["use_bias"])}}
    if "label_smoothing" in tuning_space_raw:
        hyper_obj = {**hyper_obj,
                     **{"label_smoothing": hp.choice('label_smoothing', tuning_space_raw["label_smoothing"])}}
    if "lr_decay" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"lr_decay": hp.choice('lr_decay', tuning_space_raw["lr_decay"])}}
    if "l1_flag" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"l1_flag": hp.choice('l1_flag', tuning_space_raw["l1_flag"])}}
    if "sampling" in tuning_space_raw:
        hyper_obj = {**hyper_obj, **{"sampling": hp.choice('sampling', tuning_space_raw["sampling"])}}

    return hyper_obj


def grid_search(config, run):
    # config = {
    #     "embedding_dim": [100 * i for i in range(1, 6)],
    #     "input_dropout": [0.1 * i for i in range(2)],
    #     "hidden_dropout": [0.1 * i for i in range(2)],
    # }
    from sklearn.model_selection import ParameterGrid
    for i, setting in enumerate(ParameterGrid(config)):
        # input_dropout = setting["input_dropout"]
        # hidden_dropout = setting["hidden_dropout"]
        # embedding_dim = setting["embedding_dim"]
        run(i, setting)

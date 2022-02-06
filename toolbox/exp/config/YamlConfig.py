import importlib
import os

import yaml

from pathlib import Path
from toolbox.utils.AutoML import config_tuning_space


def load_hyperparameter_yaml(config_file, hyperparams):
    if not os.path.isfile(config_file):
        raise FileNotFoundError("Cannot find configuration file %s" % config_file)
    if str(config_file).endswith("yaml") or str(config_file).endswith("yml"):
        with open(os.path.abspath(config_file), "r") as file:
            try:
                config = yaml.safe_load(file)
                algorithm = config["model_name"].lower()
                for dataset in config["datasets"]:
                    if dataset["dataset"] in hyperparams:
                        hyperparams[dataset["dataset"]][algorithm] = dataset["parameters"]
                    else:
                        hyperparams = {**hyperparams, **{dataset["dataset"]: {algorithm: dataset["parameters"]}}}
            except yaml.YAMLError:
                raise Exception("Cannot load configuration: %s" % config_file)
    else:
        raise ValueError("Configuration file must have .yaml or .yml extension: %s" % config_file)
    return hyperparams


def load_search_space_yaml(config_file, search_space):
    """ loading search space configuration from yaml file"""
    if not os.path.isfile(config_file):
        raise FileNotFoundError("Cannot find configuration file %s" % config_file)
    if str(config_file).endswith("yaml") or str(config_file).endswith("yml"):
        with open(os.path.abspath(config_file), "r") as file:
            try:
                config = yaml.safe_load(file)
                algorithm = config["model_name"].lower()
                search_space = {**search_space,
                                **{algorithm: config_tuning_space(config["search_space"])}}
            except yaml.YAMLError:
                raise Exception("Cannot load configuration: %s" % config_file)
    else:
        raise ValueError("Configuration file must have .yaml or .yml extension: %s" % config_file)
    return search_space


class YamlConfigLoader:
    """Hyper parameters loading based datasets and embedding algorithms"""

    def __init__(self, args):
        self.hyperparams = {}
        self.search_space = {}

        # load hyperparameters from options (file, dir or with pkg.)
        default_search_space_dir = Path(__file__).resolve().parent / "searchspaces"
        for config_file in default_search_space_dir.glob('**/*.yaml'):
            self.search_space = load_search_space_yaml(config_file, self.search_space)
        default_hyperparam_dir = Path(__file__).resolve().parent / "hyperparams"
        for config_file in default_hyperparam_dir.glob('**/*.yaml'):
            self.hyperparams = load_hyperparameter_yaml(config_file, self.hyperparams)

        # load search spaces from options (file, dir or with pkg.)
        if hasattr(args, "hp_abs_file") and args.hp_abs_file is not None:
            self.hyperparams = load_hyperparameter_yaml(args.hp_abs_file, self.hyperparams)
        if hasattr(args, "ss_abs_file") and args.ss_abs_file is not None:
            self.search_space = load_search_space_yaml(args.ss_abs_file, self.search_space)

    def load_hyperparameter(self, dataset_name, algorithm):
        d_name = dataset_name.lower()
        a_name = algorithm.lower()

        if d_name in self.hyperparams and a_name in self.hyperparams[d_name]:
            params = self.hyperparams[d_name][a_name]
            return params

        raise Exception("This experimental setting for (%s, %s) has not been configured" % (dataset_name, algorithm))

    def load_search_space(self, algorithm):
        if algorithm in self.search_space:
            return self.search_space[algorithm]
        raise ValueError("Hyperparameter search space is not configured for %s" % algorithm)


class Importer:
    """The class defines methods for importing pykg2vec modules.

    Importer is used to defines the maps for the algorithm names and
    provides methods for loading configuration and models.

    Attributes:
        model_path (str): Path where the models are defined.
        config_path (str): Path where the configuration for each models are defineds.
        modelMap (dict): This map transforms the names of model to the actual class names.
        configMap (dict): This map transforms the input config names to the actuall config class names.

    Examples:
        >>> from toolbox.Config import Importer
        >>> config_def, model_def = Importer().import_model_config('transe')
        >>> config = config_def()
        >>> model = model_def(config)

    """

    def __init__(self):
        self.model_path = "pykg2vec.models"
        self.config_path = "pykg2vec.config"

    def import_model_config(self, name):
        """This function imports models and configuration.

        This function is used to dynamically import the modules within pykg2vec.

        Args:
          name (str): The input to the module is either name of the model or the configuration file. The strings are converted to lowercase to makesure the user inputs can easily be matched to the names of the models and the configuration class.

        Returns:
          object: Configuration and model object after it is successfully loaded.

          `config_obj` (object): Returns the configuration class object of the corresponding algorithm.
          `model_obj` (object): Returns the model class object of the corresponding algorithm.

        Raises:
          ModuleNotFoundError: It raises a module not found error if the configuration or the model cannot be found.
        """
        config_obj = getattr(importlib.import_module(self.config_path), "Config")
        return config_obj

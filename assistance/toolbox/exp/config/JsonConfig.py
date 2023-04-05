import json
from shutil import copyfile
from typing import Any, Dict


class Config(object):
    """Config load from json file

    Examples:
        >>> from toolbox.exp.config.JsonConfig import Config
        >>> config = Config(config={"name":"hyperparams", "dataset":"FB15k"}, config_file="xxx.json")
        >>> # use it like a json object
        >>> # config.name
        >>> # config.dataset
    """

    def __init__(self, config=None, config_file=None):
        if config:
            self._update(config)
        if config_file:
            with open(config_file, 'r') as fin:
                config = json.load(fin)
            self._update(config)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __contains__(self, item):
        return item in self.__dict__

    def items(self):
        return self.__dict__.items()

    def add(self, key, value):
        """Add key value pair
        """
        self.__dict__[key] = value

    def _update(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key])
            elif isinstance(config[key], list):
                config[key] = [Config(x) if isinstance(x, dict) else x for x in config[key]]

        self.__dict__.update(config)

    def save_as_json(self, dir_name: str, filename: str):
        if type(self.source) is list:
            for s in self.source:
                c = Config(s)
                c.save(dir_name)
        elif type(self.source) is dict:
            json.dumps(self.source, indent=4)
        else:
            copyfile(self.source, dir_name + filename)

    def show(self, fun=print):
        if type(self.source) is list:
            for s in self.source:
                c = Config(s)
                c.show(fun)
        elif type(self.source) is dict:
            fun(json.dumps(self.source))
        else:
            with open(self.source) as f:
                fun(json.dumps(json.load(f), indent=4))

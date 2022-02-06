from pathlib import Path

from toolbox.utils.Log import Log


class OutputPathSchema:
    def __init__(self, output_path: Path):
        self.output_path = output_path

        self.dir_path_log = output_path / 'log'
        self.dir_path_visualize = output_path / 'visualize'
        self.dir_path_checkpoint = output_path / 'checkpoint'
        self.dir_path_deploy = output_path / 'deploy'
        self.dir_path_embedding = output_path / 'embedding'
        self.dir_path_scripts = output_path / 'scripts'

        self.build_dir_structure()

    def log_path(self, filename) -> Path:
        return self.dir_path_log / filename

    def visualize_path(self, filename) -> Path:
        return self.dir_path_visualize / filename

    def checkpoint_path(self, filename="checkpoint.tar") -> Path:
        return self.dir_path_checkpoint / filename

    def deploy_path(self, filename="model.tar") -> Path:
        return self.dir_path_deploy / filename

    def scripts_path(self, filename) -> Path:
        return self.dir_path_scripts / filename

    def embedding_path(self, filename) -> Path:
        return self.dir_path_embedding / filename

    def entity_embedding_path(self, score=-1) -> Path:
        return self.score_embedding_path("entity", score)

    def relation_embedding_path(self, score=-1) -> Path:
        return self.score_embedding_path("relation", score)

    def score_embedding_path(self, name, score=-1) -> Path:
        if score == -1:
            return self.embedding_path("%s_embedding.txt" % name)
        else:
            return self.embedding_path("%s_embedding_score_%d.txt" % (name, int(score)))

    def build_dir_structure(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.dir_path_log.mkdir(parents=True, exist_ok=True)
        self.dir_path_visualize.mkdir(parents=True, exist_ok=True)
        self.dir_path_checkpoint.mkdir(parents=True, exist_ok=True)
        self.dir_path_deploy.mkdir(parents=True, exist_ok=True)
        self.dir_path_embedding.mkdir(parents=True, exist_ok=True)
        self.dir_path_scripts.mkdir(parents=True, exist_ok=True)

    def clean(self):
        # clean the dir, and recreate dir structure
        import shutil
        shutil.rmtree(self.output_path)
        self.build_dir_structure()


class OutputSchema:
    """./output
        - experiment name
          - visualize
            - events...
          - log
            - config.log
            - loss.log
            - train.log
            - test.log
            - valid.log
          - checkpoint
            - checkpoint_score_xx.tar
          - deploy
            - model_score_xx.tar
          - embedding
            - embedding_score_xx.pkl
          - config.yaml
          - output.log

        Args:
            experiment_name (str): Name of your experiment

        Examples:
            >>> from toolbox.exp.OutputSchema import OutputSchema
            >>> output = OutputSchema("dL50a_TransE")
            >>> output.dump()

    """

    def __init__(self, experiment_name: str, overwrite=False):
        self.name = experiment_name
        self.home_path = self.output_home_path()
        self.pathSchema = OutputPathSchema(self.home_path)
        if overwrite:
            self.pathSchema.clean()
        self.logger = Log(str(self.home_path / "output.log"), name_scope=experiment_name + "output")

    def output_home_path(self) -> Path:
        data_home_path: Path = Path('.') / 'output'
        data_home_path.mkdir(parents=True, exist_ok=True)
        data_home_path = data_home_path.resolve()
        return data_home_path / self.name

    def output_path_child(self, name: str) -> Path:
        return self.home_path / name

    def child_log(self, name: str, write_to_console=False) -> Log:
        return Log(str(self.pathSchema.log_path(name)), name_scope=self.name + "output-" + name, write_to_console=write_to_console)

    def dump(self):
        """ Displays all the metadata of the knowledge graph"""
        for key, value in self.__dict__.items():
            self.logger.info("%s %s" % (key, value))

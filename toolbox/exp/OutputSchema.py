"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/10/26
@description: 输出目录管理
"""
from pathlib import Path
from typing import Union

from toolbox.utils.Log import Log


class OutputPathSchema:
    """
    输出目录 下的路径
    """

    def __init__(self, output_path: Union[Path, str]):
        self.output_path: Path = output_path if output_path is Path else Path(output_path)

        self.dir_path_log = self.output_path / 'logs'
        self.dir_path_visualize = self.output_path / 'visualize'
        self.dir_path_checkpoint = self.output_path / 'checkpoint'
        self.dir_path_latex = self.output_path / 'latex'
        self.dir_path_deploy = self.output_path / 'deploy'
        self.dir_path_scripts = self.output_path / 'scripts'

        self.build_dir_structure()

    def log_path(self, filename) -> Path:
        return self.dir_path_log / filename

    def visualize_path(self, filename) -> Path:
        return self.dir_path_visualize / filename

    def checkpoint_path(self, filename="checkpoint.tar") -> Path:
        return self.dir_path_checkpoint / filename

    def latex_path(self, filename="best.tex") -> Path:
        return self.dir_path_latex / filename

    def deploy_path(self, filename="model.tar") -> Path:
        return self.dir_path_deploy / filename

    def scripts_path(self, filename) -> Path:
        return self.dir_path_scripts / filename

    def build_dir_structure(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.dir_path_log.mkdir(parents=True, exist_ok=True)
        self.dir_path_visualize.mkdir(parents=True, exist_ok=True)
        self.dir_path_checkpoint.mkdir(parents=True, exist_ok=True)
        self.dir_path_latex.mkdir(parents=True, exist_ok=True)
        self.dir_path_deploy.mkdir(parents=True, exist_ok=True)
        self.dir_path_scripts.mkdir(parents=True, exist_ok=True)

    def clean(self):
        # clean the dir, and recreate dir structure
        import shutil
        shutil.rmtree(self.output_path)
        self.build_dir_structure()


class OutputSchema:
    """
    输出目录
      ./output
        - experiment name
          - visualize          Tensorboard 可视化
            - events...
          - logs               log 日志，包含超参数、指标、最佳指标等日志，配合 toolbox.web.log_app 使用
            - config.log
            - loss.log
          - checkpoint         检查点，用于恢复训练
            - checkpoint_score_xx.tar
          - deploy             部署模型，基于 checkpoint ，不同的是这里的 tar 文件内只包含模型，不包含优化器梯度等信息
            - model_score_xx.tar
          - config.yaml
          - output.log         打印到命令行的日志

        Args:
            experiment_name (str): Name of your experiment
            overwrite (bool): If True, it will delete the folder and create new one.

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

    def output_path_child(self, child_dir_name: str) -> Path:
        return self.home_path / child_dir_name

    def child_log(self, name: str, write_to_console=False) -> Log:
        return Log(str(self.pathSchema.log_path(name)), name_scope=self.name + "output-" + name, write_to_console=write_to_console)

    def dump(self):
        """ Displays all the metadata of the knowledge graph"""
        for key, value in self.__dict__.items():
            self.logger.info("%s %s" % (key, value))


class Cleaner:
    def __init__(self, pathSchema: OutputPathSchema):
        self.pathSchema: OutputPathSchema = pathSchema

    def remove_non_best_checkpoint_and_model(self):
        def remove_non_best(dir_path: Path):
            dir_name = str(dir_path)
            import os
            print("In", dir_name)
            filenames = os.listdir(dir_name)
            to_delete_files = set([f for f in filenames if "best" not in f])
            for filename in to_delete_files:
                print("  remove", filename)
                os.remove(str(dir_path / filename))

        remove_non_best(self.pathSchema.dir_path_checkpoint)
        remove_non_best(self.pathSchema.dir_path_deploy)

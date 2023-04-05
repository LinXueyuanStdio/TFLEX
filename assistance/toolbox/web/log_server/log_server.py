"""
@date: 2022/2/20
@description: 监听远程日志服务器
"""
import json
import os
from collections import defaultdict
from typing import List

from toolbox.exp.OutputSchema import OutputPathSchema


class LogServer:
    """
    用于读取日志的类, 用于配合Table使用
    """

    def __init__(self):
        """
        self._line_counter里面的内容:{save_log_dir: {filename: (line_count, last_change_time)}}
        """
        self._log_dir = None
        self._ignore_null_loss_or_metric = True  # 如果loss和metric都是null的话，则忽略

    def set_log_dir(self, log_dir: str):
        """
        设置 log 的存放位置
        """
        if not os.path.isdir(log_dir):
            raise RuntimeError(f"`{log_dir}` is not a valid directory.")
        self._log_dir = log_dir

    def read_logs(self, ignore_log_names: dict = None) -> List[dict]:
        """
        从日志存放路径读取日志. 只会读取有更新的log

        :param ignore_log_names: 如果包含在这个里面，就不会读取该log
        :return: 如果有内容或者有更新的内容，则返回一个 list，里面每个元素都是nested的dict.
            [{
                'id':
                'metric': {nested dict},
                'meta': {},
                ...
            },{
            }]
        """
        assert self._log_dir is not None, "You have to set log_dir first."
        if ignore_log_names is None:
            ignore_log_names = {}
        dirs = os.listdir(self._log_dir)
        logs = []
        for _dir in dirs:
            if _dir in ignore_log_names:
                continue
            dir_path = os.path.join(self._log_dir, _dir)
            if not is_valid_log_dir(dir_path):
                continue
            pathSchema = OutputPathSchema(dir_path)
            dir_path_log = str(pathSchema.dir_path_log)
            print("read", dir_path_log)
            _dict, file_stats = _read_save_log(dir_path_log, self._ignore_null_loss_or_metric)
            if len(_dict) != 0:
                logs.append({'id': _dir, **_dict})
        print(logs)
        return logs

    def read_certain_logs(self, log_dir_names: List[str]):
        """
        给定log的名称，只读取对应的log
        :param log_dir_names: list[str]
        :return: [{}, {}], nested的log
        """
        assert self._log_dir is not None, "You have to set log_dir first."
        logs = []
        for _dir in log_dir_names:
            dir_path = os.path.join(self._log_dir, _dir)
            if not os.path.isdir(dir_path):
                continue
            pathSchema = OutputPathSchema(dir_path)
            dir_path_log = str(pathSchema.dir_path_log)
            print("read", dir_path_log)
            _dict, file_stats = _read_save_log(dir_path_log, self._ignore_null_loss_or_metric)
            if len(_dict) != 0:
                logs.append({'id': _dir, **_dict})
        print(logs)
        return logs


def is_valid_log_dir(dir_path: str) -> bool:
    """
    检查dir_path是否是一个合法的log目录。合法的log目录里必须包含meta.log。

    :param dir_path: 被检测的路径
    :return: 是否合法
    """
    return os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "logs", 'meta.log'))  # 至少要有meta.log表明这个是合法的log


def _read_save_log(_save_log_dir: str, ignore_null_loss_or_metric: bool = True, file_stats: dict = None):
    """
    给定一个包含metric.log, hyper.log, meta.log以及other.log的文件夹，返回一个包含数据的dict. 如果为null则返回空字典
    不读取loss.log, 因为里面的内容对table无意义。

    :param _save_log_dir: 日志存放的目录， 已经最后一级了，即该目录下应该包含metric.log等了
    :param ignore_null_loss_or_metric: 是否忽略metric和loss都为空的文件
    :param file_stats::

            {
                'meta.log': [current_line, last_modified_time],
                'hyper.log':[], 'metric.log':[], 'other.log':[]
            }

    :return:
        _dict: {'metric': {nested dict}, 'loss': {} }
        file_stats: {'meta.log': [current_line, last_modified_time],
                     'metric.log': [, ]} # 只包含有更新的文件的内容
    """
    try:
        filenames = ['meta.log', 'hyper.log', 'best_metric.log', 'other.log']
        if file_stats is None:
            file_stats = {}
        for filename in filenames:
            if filename not in file_stats:
                file_stats[filename] = [-1, -1]
        _dict = {}

        def _is_file_empty(fn):
            empty = True
            fp = os.path.join(_save_log_dir, fn)
            if os.path.exists(fp):
                with open(fp, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(line.strip()) != 0:
                            empty = False
                            break
            return empty

        if os.path.exists(os.path.join(_save_log_dir, 'metric.log')) and \
                not os.path.exists(os.path.join(_save_log_dir, 'best_metric.log')):  # 可能是之前的版本生成的, 适配一下
            with open(os.path.join(_save_log_dir, 'metric.log'), 'r', encoding='utf-8') as f, \
                    open(os.path.join(_save_log_dir, 'best_metric.log'), 'w', encoding='utf-8') as f2:
                for line in f:
                    if not line.startswith('S'):  # 是best_metric
                        best_line = line
                        f2.write(best_line)

        empty = _is_file_empty('best_metric.log') and _is_file_empty('loss.log') and _is_file_empty('metric.log')

        if empty and ignore_null_loss_or_metric:
            return _dict, file_stats

        for filename in filenames:
            filepath = os.path.join(_save_log_dir, filename)
            last_modified_time = os.path.getmtime(filepath)
            if file_stats[filename][1] == last_modified_time:
                continue
            file_stats[filename][1] = last_modified_time
            start_line = file_stats[filename][0]
            __dict, end_line = _read_nonstep_log_file(filepath, start_line)
            file_stats[filename][0] = end_line
            _dict = merge(_dict, __dict, use_b=False)  # 在这里，需要以文件指定顺序，保留靠前的内容的值
    except Exception as e:
        print("Exception raised when read {}".format(os.path.abspath(_save_log_dir)))
        print(repr(e))
        raise e
    return _dict, file_stats


def _read_nonstep_log_file(filepath: str, start_line: int = 0) -> (dict, int):
    """
    给定一个filepath, 读取里面非Step: 开头的line，每一行为json，使用后面的内容覆盖前面的内容

    :param filepath: 读取文件的路径
    :param start_line: 从哪一行开始读取
    :return: 返回一个字典(没有内容为空)和最后读取到的行号
    """
    a = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        index = -1
        for index, line in enumerate(f):
            if index < start_line:
                continue
            if not line.startswith('S'):  # 读取非step的内容
                line = line.strip()
                try:
                    b = json.loads(line)  # TODO 如果含有非法字符(例如“!"#$%&'()*+,./:;<=>?@[\]^`{|}|~ ”)，导致前端无法显示怎么办？
                except:
                    print("Corrupted json format in {}, line:{}".format(filepath, line))
                    continue
                a = merge(a, b, use_b=True)
    return a, index + 1


def merge(a: dict, b: dict, use_b: bool = True) -> dict:
    """
    将两个dict recursive合并到a中，有相同key的，根据use_b判断使用哪个值

    :param a: 字典 a
    :param b: 字典 b
    :param use_b: 是否使用字典 b 的值
    :return: 返回字典 a
    """
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], use_b)
            elif use_b:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a
